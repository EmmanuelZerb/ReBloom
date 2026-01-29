/**
 * Health Check Service
 * Vérifie l'état de tous les services dépendants
 */

import { Redis } from 'ioredis';
import { config_ } from './config';
import { apiLogger as logger } from './logger';

// ============================================
// Types
// ============================================

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

export interface ServiceHealth {
  status: HealthStatus;
  latencyMs?: number;
  message?: string;
  lastChecked: string;
}

export interface HealthCheckResult {
  status: HealthStatus;
  timestamp: string;
  uptime: number;
  version: string;
  services: {
    redis: ServiceHealth;
    storage: ServiceHealth;
    replicate: ServiceHealth;
  };
  memory: {
    heapUsed: number;
    heapTotal: number;
    rss: number;
    external: number;
  };
}

// ============================================
// Health Checkers
// ============================================

const startTime = Date.now();

async function checkRedis(): Promise<ServiceHealth> {
  const start = Date.now();
  
  try {
    const redis = new Redis(config_.redis.url, {
      connectTimeout: 5000,
      maxRetriesPerRequest: 1,
    });
    
    await redis.ping();
    await redis.quit();
    
    return {
      status: 'healthy',
      latencyMs: Date.now() - start,
      lastChecked: new Date().toISOString(),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Connection failed';
    logger.error({ error: message }, 'Redis health check failed');
    
    return {
      status: 'unhealthy',
      latencyMs: Date.now() - start,
      message,
      lastChecked: new Date().toISOString(),
    };
  }
}

async function checkStorage(): Promise<ServiceHealth> {
  const start = Date.now();
  
  try {
    // Vérifier que le dossier de stockage est accessible
    const fs = await import('fs/promises');
    const path = await import('path');
    
    const storagePath = path.resolve(config_.storage.localPath);
    
    // Vérifier l'accès en lecture/écriture
    await fs.access(storagePath, fs.constants.R_OK | fs.constants.W_OK);
    
    // Vérifier l'espace disque disponible (simplifié)
    const stats = await fs.stat(storagePath);
    
    return {
      status: 'healthy',
      latencyMs: Date.now() - start,
      lastChecked: new Date().toISOString(),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Storage check failed';
    
    return {
      status: config_.storage.provider === 'local' ? 'unhealthy' : 'degraded',
      latencyMs: Date.now() - start,
      message,
      lastChecked: new Date().toISOString(),
    };
  }
}

async function checkReplicate(): Promise<ServiceHealth> {
  const start = Date.now();
  
  try {
    // Vérifier que le token est configuré
    if (!config_.replicate.apiToken) {
      return {
        status: 'unhealthy',
        message: 'API token not configured',
        lastChecked: new Date().toISOString(),
      };
    }
    
    // Test léger: vérifier la validité du token via l'API
    const response = await fetch('https://api.replicate.com/v1/account', {
      headers: {
        Authorization: `Token ${config_.replicate.apiToken}`,
      },
      signal: AbortSignal.timeout(5000),
    });
    
    if (!response.ok) {
      return {
        status: response.status === 401 ? 'unhealthy' : 'degraded',
        latencyMs: Date.now() - start,
        message: `API returned ${response.status}`,
        lastChecked: new Date().toISOString(),
      };
    }
    
    return {
      status: 'healthy',
      latencyMs: Date.now() - start,
      lastChecked: new Date().toISOString(),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : 'API check failed';
    
    // Si c'est un timeout ou erreur réseau, marquer comme dégradé
    return {
      status: 'degraded',
      latencyMs: Date.now() - start,
      message,
      lastChecked: new Date().toISOString(),
    };
  }
}

// ============================================
// Main Health Check
// ============================================

export async function performHealthCheck(deep = false): Promise<HealthCheckResult> {
  const [redis, storage, replicate] = await Promise.all([
    checkRedis(),
    checkStorage(),
    deep ? checkReplicate() : Promise.resolve({
      status: 'healthy' as HealthStatus,
      message: 'Skipped (shallow check)',
      lastChecked: new Date().toISOString(),
    }),
  ]);
  
  // Déterminer le statut global
  const statuses = [redis.status, storage.status, replicate.status];
  let overallStatus: HealthStatus = 'healthy';
  
  if (statuses.includes('unhealthy')) {
    // Redis ou Storage unhealthy = critique
    if (redis.status === 'unhealthy' || storage.status === 'unhealthy') {
      overallStatus = 'unhealthy';
    } else {
      overallStatus = 'degraded';
    }
  } else if (statuses.includes('degraded')) {
    overallStatus = 'degraded';
  }
  
  const memoryUsage = process.memoryUsage();
  
  return {
    status: overallStatus,
    timestamp: new Date().toISOString(),
    uptime: Math.floor((Date.now() - startTime) / 1000),
    version: process.env.npm_package_version || '1.0.0',
    services: {
      redis,
      storage,
      replicate,
    },
    memory: {
      heapUsed: Math.round(memoryUsage.heapUsed / 1024 / 1024),
      heapTotal: Math.round(memoryUsage.heapTotal / 1024 / 1024),
      rss: Math.round(memoryUsage.rss / 1024 / 1024),
      external: Math.round(memoryUsage.external / 1024 / 1024),
    },
  };
}

/**
 * Liveness check - vérifie que le serveur répond
 */
export function livenessCheck(): { status: string; timestamp: string } {
  return {
    status: 'ok',
    timestamp: new Date().toISOString(),
  };
}

/**
 * Readiness check - vérifie que le serveur peut traiter des requêtes
 */
export async function readinessCheck(): Promise<{ ready: boolean; reason?: string }> {
  try {
    const health = await performHealthCheck(false);
    
    if (health.status === 'unhealthy') {
      return {
        ready: false,
        reason: 'Critical service(s) unavailable',
      };
    }
    
    return { ready: true };
  } catch (error) {
    return {
      ready: false,
      reason: error instanceof Error ? error.message : 'Health check failed',
    };
  }
}

