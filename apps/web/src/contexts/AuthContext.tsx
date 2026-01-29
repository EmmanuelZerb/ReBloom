'use client';

/**
 * Auth Context Provider
 * Manages authentication state across the application
 */

import {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
  useRef,
  type ReactNode,
} from 'react';
import { createClient } from '@/lib/supabase/client';
import { setAuthTokenGetter } from '@/lib/api';
import type { User, Session, AuthError } from '@supabase/supabase-js';
import type { UserQuota } from '@rebloom/shared';

// ============================================
// Types
// ============================================

interface AuthState {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  error: AuthError | null;
  quota: UserQuota | null;
}

interface AuthContextValue extends AuthState {
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signUpWithEmail: (email: string, password: string) => Promise<void>;
  signInWithGoogle: () => Promise<void>;
  signInWithGithub: () => Promise<void>;
  signOut: () => Promise<void>;
  refreshQuota: () => Promise<void>;
  getAccessToken: () => Promise<string | null>;
}

// ============================================
// Context
// ============================================

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

// ============================================
// Provider
// ============================================

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    session: null,
    isLoading: true,
    error: null,
    quota: null,
  });

  const supabase = createClient();
  const tokenGetterInitialized = useRef(false);

  // Initialize auth state
  useEffect(() => {
    // Initialize API token getter (only once)
    if (!tokenGetterInitialized.current) {
      setAuthTokenGetter(async () => {
        const { data: { session } } = await supabase.auth.getSession();
        return session?.access_token ?? null;
      });
      tokenGetterInitialized.current = true;
    }
    const initAuth = async () => {
      try {
        const { data: { session }, error } = await supabase.auth.getSession();

        if (error) {
          setState((s) => ({ ...s, error, isLoading: false }));
          return;
        }

        setState((s) => ({
          ...s,
          user: session?.user ?? null,
          session,
          isLoading: false,
        }));
      } catch (error) {
        console.error('Auth init error:', error);
        setState((s) => ({ ...s, isLoading: false }));
      }
    };

    initAuth();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        setState((s) => ({
          ...s,
          user: session?.user ?? null,
          session,
          isLoading: false,
        }));

        // Refresh quota when user logs in
        if (event === 'SIGNED_IN' && session) {
          refreshQuota();
        }
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, []);

  // Get access token for API calls
  const getAccessToken = useCallback(async (): Promise<string | null> => {
    const { data: { session } } = await supabase.auth.getSession();
    return session?.access_token ?? null;
  }, [supabase]);

  // Refresh user quota
  const refreshQuota = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/api/auth/quota`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setState((s) => ({ ...s, quota: data.quota }));
      }
    } catch (error) {
      console.error('Failed to refresh quota:', error);
    }
  }, [getAccessToken]);

  // Sign in with email/password
  const signInWithEmail = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, isLoading: true, error: null }));

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (error) {
      setState((s) => ({ ...s, error, isLoading: false }));
      throw error;
    }

    setState((s) => ({ ...s, isLoading: false }));
  }, [supabase]);

  // Sign up with email/password
  const signUpWithEmail = useCallback(async (email: string, password: string) => {
    setState((s) => ({ ...s, isLoading: true, error: null }));

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) {
      setState((s) => ({ ...s, error, isLoading: false }));
      throw error;
    }

    setState((s) => ({ ...s, isLoading: false }));
  }, [supabase]);

  // Sign in with Google
  const signInWithGoogle = useCallback(async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) throw error;
  }, [supabase]);

  // Sign in with GitHub
  const signInWithGithub = useCallback(async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'github',
      options: {
        redirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error) throw error;
  }, [supabase]);

  // Sign out
  const signOut = useCallback(async () => {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;

    setState((s) => ({
      ...s,
      user: null,
      session: null,
      quota: null,
    }));
  }, [supabase]);

  const value: AuthContextValue = {
    ...state,
    signInWithEmail,
    signUpWithEmail,
    signInWithGoogle,
    signInWithGithub,
    signOut,
    refreshQuota,
    getAccessToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

// ============================================
// Hook
// ============================================

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }

  return context;
}
