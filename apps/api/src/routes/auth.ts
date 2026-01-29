/**
 * Auth Routes
 * Provides user information and session validation endpoints
 * Note: Actual auth is handled by Supabase on the frontend
 */

import { Hono } from 'hono';
import { requireAuth, optionalAuth } from '../lib/auth';
import { getUserById, getUserQuota, isUserPremium } from '../lib/supabase';
import { apiLogger as logger } from '../lib/logger';

const auth = new Hono();

/**
 * GET /api/auth/me
 * Get current authenticated user info
 */
auth.get('/me', requireAuth(), async (c) => {
  const userId = c.get('userId');
  const user = c.get('user');

  // Get additional user info from database if needed
  const [quota, isPremium] = await Promise.all([
    getUserQuota(userId),
    isUserPremium(userId),
  ]);

  return c.json({
    success: true,
    user: {
      id: userId,
      email: user.email,
      subscriptionTier: isPremium ? 'premium' : 'free',
    },
    quota,
  });
});

/**
 * GET /api/auth/quota
 * Get current user's quota information
 */
auth.get('/quota', requireAuth(), async (c) => {
  const userId = c.get('userId');
  const quota = await getUserQuota(userId);

  return c.json({
    success: true,
    quota,
  });
});

/**
 * GET /api/auth/validate
 * Validate that the current token is valid
 * Useful for checking session status without fetching full user info
 */
auth.get('/validate', requireAuth(), async (c) => {
  const userId = c.get('userId');

  return c.json({
    success: true,
    valid: true,
    userId,
  });
});

/**
 * GET /api/auth/status
 * Check auth status (works for both authenticated and anonymous users)
 */
auth.get('/status', optionalAuth(), async (c) => {
  const userId = c.get('userId');

  if (userId) {
    return c.json({
      success: true,
      authenticated: true,
      userId,
    });
  }

  return c.json({
    success: true,
    authenticated: false,
  });
});

export { auth };
