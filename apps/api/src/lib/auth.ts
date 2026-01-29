/**
 * Authentication Middleware
 * Validates Supabase JWT tokens and protects routes
 */

import { Context, Next } from 'hono';
import jwt from 'jsonwebtoken';
import { config_ } from './config';
import { apiLogger as logger } from './logger';
import { UnauthorizedError } from './errors';
import { getUserQuota, incrementUserUsage, type AuthUser } from './supabase';

// ============================================
// Types
// ============================================

interface JWTPayload {
  sub: string;           // User ID
  email?: string;
  role?: string;
  aud?: string;
  exp?: number;
  iat?: number;
}

// Extend Hono's context to include user
declare module 'hono' {
  interface ContextVariableMap {
    user: AuthUser;
    userId: string;
  }
}

// ============================================
// JWT Verification
// ============================================

/**
 * Verify and decode a Supabase JWT token
 */
function verifyToken(token: string): JWTPayload | null {
  try {
    const decoded = jwt.verify(token, config_.supabase.jwtSecret, {
      algorithms: ['HS256'],
    }) as JWTPayload;

    return decoded;
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      logger.debug('Token expired');
    } else if (error instanceof jwt.JsonWebTokenError) {
      logger.debug({ error: (error as Error).message }, 'Invalid token');
    }
    return null;
  }
}

/**
 * Extract Bearer token from Authorization header
 */
function extractToken(authHeader: string | undefined): string | null {
  if (!authHeader) return null;

  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0].toLowerCase() !== 'bearer') {
    return null;
  }

  return parts[1];
}

// ============================================
// Authentication Middleware
// ============================================

/**
 * Require authentication for protected routes
 * Validates JWT and adds user to context
 */
export function requireAuth() {
  return async (c: Context, next: Next) => {
    // Skip auth in development if AUTH_REQUIRED is false
    if (!config_.auth.required && config_.isDev) {
      logger.debug('Auth skipped (AUTH_REQUIRED=false in dev mode)');
      c.set('user', { id: 'dev-user', email: 'dev@example.com' });
      c.set('userId', 'dev-user');
      await next();
      return;
    }

    const authHeader = c.req.header('Authorization');
    const token = extractToken(authHeader);

    if (!token) {
      logger.debug('No token provided');
      throw new UnauthorizedError('Authentication required');
    }

    const payload = verifyToken(token);

    if (!payload || !payload.sub) {
      logger.warn('Invalid or expired token');
      throw new UnauthorizedError('Invalid or expired token');
    }

    // Add user to context
    const user: AuthUser = {
      id: payload.sub,
      email: payload.email,
      role: payload.role,
      aud: payload.aud,
    };

    c.set('user', user);
    c.set('userId', payload.sub);

    logger.debug({ userId: payload.sub }, 'User authenticated');

    await next();
  };
}

/**
 * Optional authentication - doesn't fail if no token
 * Useful for routes that work for both authenticated and anonymous users
 */
export function optionalAuth() {
  return async (c: Context, next: Next) => {
    const authHeader = c.req.header('Authorization');
    const token = extractToken(authHeader);

    if (token) {
      const payload = verifyToken(token);

      if (payload?.sub) {
        const user: AuthUser = {
          id: payload.sub,
          email: payload.email,
          role: payload.role,
          aud: payload.aud,
        };

        c.set('user', user);
        c.set('userId', payload.sub);
        logger.debug({ userId: payload.sub }, 'User authenticated (optional)');
      }
    }

    await next();
  };
}

/**
 * Check user quota before allowing upload
 * Returns 429 if quota exceeded
 */
export function checkQuota() {
  return async (c: Context, next: Next) => {
    const userId = c.get('userId');

    if (!userId) {
      // Anonymous users get very limited quota via rate limiting
      await next();
      return;
    }

    const quota = await getUserQuota(userId);

    if (quota.used >= quota.limit) {
      logger.warn({ userId, used: quota.used, limit: quota.limit }, 'User quota exceeded');

      return c.json(
        {
          success: false,
          error: {
            code: 'QUOTA_EXCEEDED',
            message: `Daily quota exceeded. You have used ${quota.used}/${quota.limit} images today.`,
            resetAt: quota.resetAt,
          },
        },
        429
      );
    }

    // Add quota info to response headers
    c.header('X-Quota-Used', quota.used.toString());
    c.header('X-Quota-Limit', quota.limit.toString());
    c.header('X-Quota-Reset', quota.resetAt);

    await next();
  };
}

/**
 * Increment usage after successful upload
 * Call this after the upload is processed
 */
export async function recordUsage(userId: string): Promise<void> {
  if (!userId || userId === 'dev-user') return;

  await incrementUserUsage(userId);
  logger.debug({ userId }, 'Usage incremented');
}

// ============================================
// Role-based Access Control
// ============================================

/**
 * Require specific role(s) for access
 */
export function requireRole(...roles: string[]) {
  return async (c: Context, next: Next) => {
    const user = c.get('user');

    if (!user) {
      throw new UnauthorizedError('Authentication required');
    }

    if (!user.role || !roles.includes(user.role)) {
      logger.warn({ userId: user.id, role: user.role, required: roles }, 'Insufficient role');
      throw new UnauthorizedError('Insufficient permissions');
    }

    await next();
  };
}

/**
 * Require admin role
 */
export const requireAdmin = () => requireRole('service_role', 'admin');
