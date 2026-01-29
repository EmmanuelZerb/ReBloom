'use client';

/**
 * User Menu Component
 * Shows user info and logout button when authenticated
 */

import { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { User, LogOut, Settings, ChevronDown, Loader2 } from 'lucide-react';

export function UserMenu() {
  const { user, quota, signOut, isLoading } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [isSigningOut, setIsSigningOut] = useState(false);

  const handleSignOut = async () => {
    setIsSigningOut(true);
    try {
      await signOut();
    } catch (error) {
      console.error('Sign out error:', error);
    } finally {
      setIsSigningOut(false);
    }
  };

  if (isLoading) {
    return (
      <div className="h-10 w-10 rounded-full bg-gray-200 animate-pulse" />
    );
  }

  if (!user) {
    return (
      <a
        href="/auth/login"
        className="px-4 py-2 text-sm font-medium text-white bg-coral-500 hover:bg-coral-600 rounded-lg transition-colors"
      >
        Sign in
      </a>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
      >
        <div className="h-8 w-8 rounded-full bg-coral-100 flex items-center justify-center">
          {user.user_metadata?.avatar_url ? (
            <img
              src={user.user_metadata.avatar_url}
              alt="Avatar"
              className="h-8 w-8 rounded-full"
            />
          ) : (
            <User className="h-4 w-4 text-coral-600" />
          )}
        </div>
        <span className="text-sm font-medium text-gray-700 hidden sm:block">
          {user.email?.split('@')[0]}
        </span>
        <ChevronDown className="h-4 w-4 text-gray-500" />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-20">
            <div className="px-4 py-2 border-b border-gray-100">
              <p className="text-sm font-medium text-gray-900">{user.email}</p>
              {quota && (
                <p className="text-xs text-gray-500 mt-1">
                  {quota.used}/{quota.limit} images today
                </p>
              )}
            </div>

            <div className="py-1">
              <a
                href="/settings"
                className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
              >
                <Settings className="h-4 w-4" />
                Settings
              </a>

              <button
                onClick={handleSignOut}
                disabled={isSigningOut}
                className="w-full flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-red-50"
              >
                {isSigningOut ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <LogOut className="h-4 w-4" />
                )}
                Sign out
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
