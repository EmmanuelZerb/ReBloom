-- =============================================
-- ReBloom Supabase Database Schema
-- =============================================
-- Run this in your Supabase SQL Editor to set up the required tables

-- =============================================
-- Profiles Table (extends auth.users)
-- =============================================
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  email TEXT,
  name TEXT,
  avatar_url TEXT,
  subscription_tier TEXT DEFAULT 'free' CHECK (subscription_tier IN ('free', 'premium', 'pro')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Policies for profiles
CREATE POLICY "Users can view their own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Trigger to auto-create profile on user signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, name, avatar_url)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'full_name', NEW.raw_user_meta_data->>'name'),
    COALESCE(NEW.raw_user_meta_data->>'avatar_url', NEW.raw_user_meta_data->>'picture')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create trigger for new user signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- =============================================
-- User Quotas Table
-- =============================================
CREATE TABLE IF NOT EXISTS public.user_quotas (
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  images_processed INTEGER DEFAULT 0,
  quota_limit INTEGER DEFAULT 10,
  reset_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '1 day'),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS
ALTER TABLE public.user_quotas ENABLE ROW LEVEL SECURITY;

-- Policies for user_quotas
CREATE POLICY "Users can view their own quota"
  ON public.user_quotas FOR SELECT
  USING (auth.uid() = user_id);

-- Service role can manage all quotas
CREATE POLICY "Service role can manage quotas"
  ON public.user_quotas FOR ALL
  USING (auth.role() = 'service_role');

-- Function to increment user usage
CREATE OR REPLACE FUNCTION public.increment_user_usage(p_user_id UUID)
RETURNS VOID AS $$
DECLARE
  current_reset TIMESTAMPTZ;
BEGIN
  -- Get current reset time
  SELECT reset_at INTO current_reset FROM public.user_quotas WHERE user_id = p_user_id;

  -- If no record exists, create one
  IF current_reset IS NULL THEN
    INSERT INTO public.user_quotas (user_id, images_processed, reset_at)
    VALUES (p_user_id, 1, NOW() + INTERVAL '1 day');
    RETURN;
  END IF;

  -- If reset time has passed, reset counter
  IF current_reset < NOW() THEN
    UPDATE public.user_quotas
    SET images_processed = 1,
        reset_at = NOW() + INTERVAL '1 day',
        updated_at = NOW()
    WHERE user_id = p_user_id;
    RETURN;
  END IF;

  -- Otherwise, increment counter
  UPDATE public.user_quotas
  SET images_processed = images_processed + 1,
      updated_at = NOW()
  WHERE user_id = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =============================================
-- Jobs Table (optional - for job history)
-- =============================================
CREATE TABLE IF NOT EXISTS public.jobs (
  id TEXT PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
  progress INTEGER DEFAULT 0,
  original_filename TEXT,
  original_size INTEGER,
  processed_filename TEXT,
  processed_size INTEGER,
  error_message TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

-- Enable RLS
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

-- Policies for jobs
CREATE POLICY "Users can view their own jobs"
  ON public.jobs FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own jobs"
  ON public.jobs FOR DELETE
  USING (auth.uid() = user_id);

-- Service role can manage all jobs
CREATE POLICY "Service role can manage jobs"
  ON public.jobs FOR ALL
  USING (auth.role() = 'service_role');

-- =============================================
-- Indexes
-- =============================================
CREATE INDEX IF NOT EXISTS idx_profiles_email ON public.profiles(email);
CREATE INDEX IF NOT EXISTS idx_user_quotas_reset_at ON public.user_quotas(reset_at);
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON public.jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON public.jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON public.jobs(status);

-- =============================================
-- Updated_at Trigger Function
-- =============================================
CREATE OR REPLACE FUNCTION public.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
CREATE TRIGGER set_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER set_user_quotas_updated_at
  BEFORE UPDATE ON public.user_quotas
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

CREATE TRIGGER set_jobs_updated_at
  BEFORE UPDATE ON public.jobs
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- =============================================
-- Grant Permissions
-- =============================================
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;
