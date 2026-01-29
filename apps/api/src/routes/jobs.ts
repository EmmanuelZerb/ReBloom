/**
 * Routes de gestion des jobs
 * GET /api/jobs/:id - Statut d'un job
 * GET /api/jobs/:id/result - Résultat d'un job terminé
 * POST /api/webhooks/replicate - Webhook Replicate (secured with HMAC)
 */

import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { jobManager } from '../services/queue';
import { storage } from '../services/storage';
import { apiLogger as logger } from '../lib/logger';
import { jobIdSchema, replicateWebhookSchema } from '../lib/validation';
import { JobNotFoundError, AppError } from '../lib/errors';
import { validateWebhookSignature } from '../lib/webhook';
import { ERROR_CODES } from '@rebloom/shared';
import type { JobStatusResponse, DownloadResponse } from '@rebloom/shared';
import { generateDownloadToken } from '../lib/signed-urls';

// Type augmentation for Hono context
declare module 'hono' {
  interface ContextVariableMap {
    webhookPayload: unknown;
  }
}

const jobs = new Hono();

/**
 * GET /api/jobs/:id
 * Récupère le statut d'un job
 */
jobs.get('/:id', zValidator('param', jobIdSchema), async (c) => {
  const { id } = c.req.valid('param');

  const job = await jobManager.get(id);
  if (!job) {
    throw new JobNotFoundError(id);
  }

  const response: JobStatusResponse = {
    success: true,
    job,
  };

  return c.json(response);
});

/**
 * GET /api/jobs/:id/result
 * Récupère le résultat d'un job terminé (image traitée)
 * Returns a signed download URL for secure access
 */
jobs.get('/:id/result', zValidator('param', jobIdSchema), async (c) => {
  const { id } = c.req.valid('param');

  const job = await jobManager.get(id);
  if (!job) {
    throw new JobNotFoundError(id);
  }

  if (job.status !== 'completed') {
    throw new AppError(
      ERROR_CODES.JOB_NOT_FOUND,
      `Job is not completed yet. Current status: ${job.status}`,
      400,
      { status: job.status }
    );
  }

  if (!job.processedImage) {
    throw new AppError(ERROR_CODES.PROCESSING_FAILED, 'Processed image not found', 500);
  }

  // Generate signed download token (valid for 24 hours)
  const expirySeconds = 24 * 60 * 60;
  const token = generateDownloadToken(id, expirySeconds);
  const downloadUrl = `/api/download/${id}?token=${token}`;

  const response: DownloadResponse = {
    success: true,
    downloadUrl,
    expiresAt: new Date(Date.now() + expirySeconds * 1000).toISOString(),
  };

  return c.json(response);
});

/**
 * POST /api/webhooks/replicate
 * Webhook appelé par Replicate quand le traitement est terminé
 * Secured with HMAC-SHA256 signature validation
 */
jobs.post('/webhooks/replicate', validateWebhookSignature(), async (c) => {
  // Payload is pre-validated and stored by the middleware
  const rawPayload = c.get('webhookPayload');

  // Validate payload structure with Zod
  const parseResult = replicateWebhookSchema.safeParse(rawPayload);
  if (!parseResult.success) {
    logger.warn({ errors: parseResult.error.errors }, 'Invalid webhook payload structure');
    return c.json({ received: false, error: 'Invalid payload' }, 400);
  }

  const payload = parseResult.data;

  logger.info({ predictionId: payload.id, status: payload.status }, 'Replicate webhook received (verified)');

  // Le predictionId doit correspondre à un jobId dans notre système
  // Pour l'instant, on gère ça dans le worker
  // Ce webhook est pour les traitements async futurs

  if (payload.status === 'succeeded' && payload.output) {
    const outputUrl = Array.isArray(payload.output) ? payload.output[0] : payload.output;
    logger.info({ outputUrl }, 'Processing succeeded via webhook');
  }

  if (payload.status === 'failed') {
    logger.error({ error: payload.error }, 'Processing failed via webhook');
  }

  return c.json({ received: true });
});

export { jobs };
