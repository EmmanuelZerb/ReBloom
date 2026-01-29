/**
 * Rate Limiting Middleware
 * Implémentation robuste avec support Redis et fallback en mémoire
 */

import { Context, Next } from 'hono';
import { Redis } from 'ioredis';
import { config_ } from './config';
import { apiLogger as logger } from './logger';

// ============================================
// Types
// ============================================

interface RateLimitConfig {
  /** Nombre de requêtes maximum */
  maxRequests: number;
  /** Fenêtre de temps en secondes */
  windowSeconds: number;
  /** Préfixe pour les clés Redis */
  keyPrefix?: string;
  /** Message d'erreur personnalisé */
  message?: string;
  /** Fonction pour extraire l'identifiant (IP par défaut) */
  keyGenerator?: (c: Context) => string;
}

interface RateLimitResult {
  success: boolean;
  limit: number;
  remaining: number;
  reset: number;
}

// ============================================
// In-Memory Store (Fallback)
// ============================================

class MemoryStore {
  private store: Map<string, { count: number; resetAt: number }> = new Map();

  async increment(key: string, windowSeconds: number): Promise<RateLimitResult> {
    const now = Date.now();
    const resetAt = now + windowSeconds * 1000;
    
    const existing = this.store.get(key);
    
    if (existing && existing.resetAt > now) {
      existing.count++;
      return {
        success: true,
        limit: 0,
        remaining: 0,
        reset: Math.ceil((existing.resetAt - now) / 1000),
      };
    }
    
    this.store.set(key, { count: 1, resetAt });
    
    // Nettoyage périodique
    if (this.store.size > 10000) {
      this.cleanup();
    }
    
    return {
      success: true,
      limit: 0,
      remaining: 0,
      reset: windowSeconds,
    };
  }

  async getCount(key: string): Promise<number> {
    const existing = this.store.get(key);
    if (!existing || existing.resetAt <= Date.now()) {
      return 0;
    }
    return existing.count;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, value] of this.store.entries()) {
      if (value.resetAt <= now) {
        this.store.delete(key);
      }
    }
  }
}

// ============================================
// Redis Store
// ============================================

class RedisStore {
  private redis: Redis;

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl, {
      maxRetriesPerRequest: 3,
      retryDelayOnFailover: 100,
    });
  }

  async increment(key: string, windowSeconds: number): Promise<RateLimitResult> {
    const multi = this.redis.multi();
    multi.incr(key);
    multi.ttl(key);
    
    const results = await multi.exec();
    
    if (!results) {
      throw new Error('Redis transaction failed');
    }
    
    const count = results[0][1] as number;
    let ttl = results[1][1] as number;
    
    // Si la clé est nouvelle, définir le TTL
    if (ttl === -1) {
      await this.redis.expire(key, windowSeconds);
      ttl = windowSeconds;
    }
    
    return {
      success: true,
      limit: 0,
      remaining: 0,
      reset: ttl,
    };
  }

  async getCount(key: string): Promise<number> {
    const count = await this.redis.get(key);
    return count ? parseInt(count, 10) : 0;
  }

  async close(): Promise<void> {
    await this.redis.quit();
  }
}

// ============================================
// Rate Limiter Factory
// ============================================

let redisStore: RedisStore | null = null;
const memoryStore = new MemoryStore();

function getStore(): RedisStore | MemoryStore {
  if (redisStore) return redisStore;
  
  try {
    redisStore = new RedisStore(config_.redis.url);
    logger.info('Rate limiter using Redis store');
    return redisStore;
  } catch (error) {
    logger.warn('Redis unavailable, using in-memory rate limiting');
    return memoryStore;
  }
}

// ============================================
// Rate Limit Middleware
// ============================================

const defaultKeyGenerator = (c: Context): string => {
  // Essayer de récupérer l'IP réelle derrière un proxy
  const forwarded = c.req.header('x-forwarded-for');
  const realIp = c.req.header('x-real-ip');
  const cfConnectingIp = c.req.header('cf-connecting-ip');
  
  return cfConnectingIp || realIp || forwarded?.split(',')[0] || 'unknown';
};

export function rateLimit(config: RateLimitConfig) {
  const {
    maxRequests,
    windowSeconds,
    keyPrefix = 'rl',
    message = 'Too many requests, please try again later',
    keyGenerator = defaultKeyGenerator,
  } = config;

  return async (c: Context, next: Next) => {
    const store = getStore();
    const identifier = keyGenerator(c);
    const key = `${keyPrefix}:${identifier}`;

    try {
      const result = await store.increment(key, windowSeconds);
      const count = await store.getCount(key);
      const remaining = Math.max(0, maxRequests - count);

      // Ajouter les headers de rate limit
      c.header('X-RateLimit-Limit', maxRequests.toString());
      c.header('X-RateLimit-Remaining', remaining.toString());
      c.header('X-RateLimit-Reset', result.reset.toString());

      if (count > maxRequests) {
        logger.warn({ identifier, count, maxRequests }, 'Rate limit exceeded');
        
        c.header('Retry-After', result.reset.toString());
        
        return c.json(
          {
            success: false,
            error: {
              code: 'RATE_LIMIT_EXCEEDED',
              message,
              retryAfter: result.reset,
            },
          },
          429
        );
      }

      await next();
    } catch (error) {
      // En cas d'erreur, laisser passer la requête (fail-open)
      logger.error({ error }, 'Rate limit check failed');
      await next();
    }
  };
}

// ============================================
// Preset Rate Limiters
// ============================================

/** Limite stricte pour l'upload (10 requêtes par minute) */
export const uploadRateLimit = rateLimit({
  maxRequests: 10,
  windowSeconds: 60,
  keyPrefix: 'rl:upload',
  message: 'Upload limit reached. Please wait before uploading more images.',
});

/** Limite standard pour les API (100 requêtes par minute) */
export const apiRateLimit = rateLimit({
  maxRequests: 100,
  windowSeconds: 60,
  keyPrefix: 'rl:api',
});

/** Limite pour le polling de statut (300 requêtes par minute) */
export const pollingRateLimit = rateLimit({
  maxRequests: 300,
  windowSeconds: 60,
  keyPrefix: 'rl:poll',
});

/** Limite globale par IP (1000 requêtes par minute) */
export const globalRateLimit = rateLimit({
  maxRequests: 1000,
  windowSeconds: 60,
  keyPrefix: 'rl:global',
});

