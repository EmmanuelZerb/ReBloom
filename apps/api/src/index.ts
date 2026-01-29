/**
 * ReBloom API Server
 * Point d'entrée principal de l'application
 */

import { serve } from '@hono/node-server';
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger as honoLogger } from 'hono/logger';
import { prettyJSON } from 'hono/pretty-json';
import { secureHeaders } from 'hono/secure-headers';
import { timing } from 'hono/timing';
import { compress } from 'hono/compress';

import { config_ } from './lib/config';
import { apiLogger as logger, createLogger } from './lib/logger';
import { AppError } from './lib/errors';
import { ERROR_CODES } from '@rebloom/shared';
import { globalRateLimit } from './lib/rate-limit';
import { performHealthCheck, livenessCheck, readinessCheck } from './lib/health';

// Routes
import { upload } from './routes/upload';
import { jobs } from './routes/jobs';
import { download } from './routes/download';

// Worker (import pour démarrer)
import './workers/processor';

// ============================================
// App Setup
// ============================================

const app = new Hono();

// ============================================
// Middlewares
// ============================================

// Compression
app.use('*', compress());

// Server Timing headers (dev only)
if (config_.isDev) {
  app.use('*', timing());
}

// CORS
app.use(
  '*',
  cors({
    origin: config_.isDev 
      ? '*' 
      : ['http://localhost:3000', 'https://rebloom.app'],
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
    exposeHeaders: [
      'Content-Length', 
      'Content-Disposition',
      'X-RateLimit-Limit',
      'X-RateLimit-Remaining',
      'X-RateLimit-Reset',
    ],
    maxAge: 86400,
    credentials: true,
  })
);

// Enhanced security headers
app.use('*', secureHeaders({
  contentSecurityPolicy: config_.isProd ? {
    defaultSrc: ["'self'"],
    scriptSrc: ["'self'"],
    styleSrc: ["'self'", "'unsafe-inline'"],
    imgSrc: ["'self'", 'data:', 'blob:', 'https:'],
    connectSrc: ["'self'", 'https://api.replicate.com'],
    frameSrc: ["'none'"],
    objectSrc: ["'none'"],
  } : undefined,
  xFrameOptions: 'DENY',
  xContentTypeOptions: 'nosniff',
  referrerPolicy: 'strict-origin-when-cross-origin',
}));

// Global rate limiting
app.use('*', globalRateLimit);

// Request ID middleware
app.use('*', async (c, next) => {
  const requestId = c.req.header('x-request-id') || crypto.randomUUID();
  c.header('X-Request-ID', requestId);
  c.set('requestId', requestId);
  await next();
});

// Pretty JSON in dev
if (config_.isDev) {
  app.use('*', prettyJSON());
}

// Request logging with structured data
app.use('*', async (c, next) => {
  const start = Date.now();
  const requestId = c.get('requestId');
  
  await next();
  
  const ms = Date.now() - start;
  const logData = {
    requestId,
    method: c.req.method,
    path: c.req.path,
    status: c.res.status,
    duration: ms,
    userAgent: c.req.header('user-agent')?.substring(0, 100),
  };
  
  if (c.res.status >= 400) {
    logger.warn(logData, 'Request completed with error');
  } else {
    logger.info(logData, 'Request completed');
  }
});

// ============================================
// Routes
// ============================================

// Health endpoints (Kubernetes-style)
app.get('/health', async (c) => {
  const deep = c.req.query('deep') === 'true';
  const health = await performHealthCheck(deep);
  
  const statusCode = health.status === 'healthy' ? 200 
    : health.status === 'degraded' ? 200 
    : 503;
  
  return c.json(health, statusCode);
});

// Liveness probe - is the server alive?
app.get('/health/live', (c) => {
  return c.json(livenessCheck());
});

// Readiness probe - can the server handle requests?
app.get('/health/ready', async (c) => {
  const result = await readinessCheck();
  return c.json(result, result.ready ? 200 : 503);
});

// API routes
app.route('/api/upload', upload);
app.route('/api/jobs', jobs);
app.route('/api/download', download);
app.route('/api', download); // Pour /api/files/*

// ============================================
// Error Handling
// ============================================

app.onError((err, c) => {
  logger.error({ error: err.message, stack: err.stack }, 'Request error');

  if (err instanceof AppError) {
    return c.json(err.toJSON(), err.statusCode as any);
  }

  // Erreur Zod (validation)
  if (err.name === 'ZodError') {
    return c.json(
      {
        success: false,
        error: {
          code: ERROR_CODES.INVALID_FILE_TYPE,
          message: 'Validation failed',
          details: (err as any).errors,
        },
      },
      400
    );
  }

  // Erreur générique
  return c.json(
    {
      success: false,
      error: {
        code: ERROR_CODES.INTERNAL_ERROR,
        message: config_.isDev ? err.message : 'Internal server error',
      },
    },
    500
  );
});

// 404
app.notFound((c) => {
  return c.json(
    {
      success: false,
      error: {
        code: ERROR_CODES.INTERNAL_ERROR,
        message: 'Route not found',
      },
    },
    404
  );
});

// ============================================
// Server Start
// ============================================

const server = serve(
  {
    fetch: app.fetch,
    port: config_.server.port,
    hostname: config_.server.host,
  },
  (info) => {
    logger.info(`
╔═══════════════════════════════════════════════╗
║                                               ║
║   🌸 ReBloom API Server                       ║
║                                               ║
║   Running on: http://${info.address}:${info.port}        ║
║   Environment: ${config_.isDev ? 'development' : 'production'}                  ║
║                                               ║
╚═══════════════════════════════════════════════╝
    `);
  }
);

// ============================================
// Graceful Shutdown
// ============================================

async function shutdown() {
  logger.info('Shutting down server...');
  server.close();
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

export default app;
