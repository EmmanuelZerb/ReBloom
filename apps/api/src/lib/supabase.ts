/**
 * Supabase Client Configuration
 * Provides both anon and service role clients for different use cases
 */

import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { config_ } from './config';
import { apiLogger as logger } from './logger';

// ============================================
// Supabase Clients
// ============================================

/**
 * Supabase client with anon key (public operations)
 * Use this for client-facing operations that respect RLS
 */
export const supabaseAnon: SupabaseClient = createClient(
  config_.supabase.url,
  config_.supabase.anonKey,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  }
);

/**
 * Supabase client with service role key (admin operations)
 * Use this for server-side operations that bypass RLS
 * WARNING: Never expose this client to the frontend
 */
export const supabaseAdmin: SupabaseClient = createClient(
  config_.supabase.url,
  config_.supabase.serviceRoleKey,
  {
    auth: {
      autoRefreshToken: false,
      persistSession: false,
    },
  }
);

// ============================================
// User Types
// ============================================

export interface AuthUser {
  id: string;
  email?: string;
  role?: string;
  aud?: string;
}

// ============================================
// Helper Functions
// ============================================

/**
 * Get user from Supabase by their ID
 */
export async function getUserById(userId: string): Promise<AuthUser | null> {
  try {
    const { data, error } = await supabaseAdmin.auth.admin.getUserById(userId);

    if (error || !data.user) {
      logger.warn({ userId, error }, 'Failed to get user by ID');
      return null;
    }

    return {
      id: data.user.id,
      email: data.user.email,
      role: data.user.role,
      aud: data.user.aud,
    };
  } catch (error) {
    logger.error({ userId, error }, 'Error fetching user');
    return null;
  }
}

/**
 * Check if user has premium/paid subscription
 * This checks the user's metadata or a profiles table
 */
export async function isUserPremium(userId: string): Promise<boolean> {
  try {
    // Option 1: Check user metadata
    const { data, error } = await supabaseAdmin
      .from('profiles')
      .select('subscription_tier')
      .eq('id', userId)
      .single();

    if (error) {
      // If profiles table doesn't exist, default to non-premium
      logger.debug({ userId }, 'No profile found, assuming free tier');
      return false;
    }

    return data?.subscription_tier === 'premium' || data?.subscription_tier === 'pro';
  } catch (error) {
    logger.error({ userId, error }, 'Error checking premium status');
    return false;
  }
}

/**
 * Get user's usage quota from database
 */
export async function getUserQuota(userId: string): Promise<{
  used: number;
  limit: number;
  resetAt: string;
}> {
  try {
    const { data, error } = await supabaseAdmin
      .from('user_quotas')
      .select('images_processed, quota_limit, reset_at')
      .eq('user_id', userId)
      .single();

    if (error || !data) {
      // Default quota for new users
      return {
        used: 0,
        limit: 10, // Free tier: 10 images per day
        resetAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(),
      };
    }

    return {
      used: data.images_processed || 0,
      limit: data.quota_limit || 10,
      resetAt: data.reset_at,
    };
  } catch (error) {
    logger.error({ userId, error }, 'Error fetching user quota');
    return { used: 0, limit: 10, resetAt: new Date().toISOString() };
  }
}

/**
 * Increment user's usage counter
 */
export async function incrementUserUsage(userId: string): Promise<boolean> {
  try {
    const { error } = await supabaseAdmin.rpc('increment_user_usage', {
      p_user_id: userId,
    });

    if (error) {
      // Fallback: try direct update
      const { error: updateError } = await supabaseAdmin
        .from('user_quotas')
        .upsert({
          user_id: userId,
          images_processed: 1,
          updated_at: new Date().toISOString(),
        }, {
          onConflict: 'user_id',
        });

      if (updateError) {
        logger.error({ userId, error: updateError }, 'Failed to increment usage');
        return false;
      }
    }

    return true;
  } catch (error) {
    logger.error({ userId, error }, 'Error incrementing usage');
    return false;
  }
}

logger.info({ url: config_.supabase.url }, 'Supabase client initialized');
