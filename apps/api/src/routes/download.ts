/**
 * Routes de téléchargement
 * GET /api/download/:id - Télécharger une image traitée
 * GET /api/files/* - Servir les fichiers statiques (dev)
 */

import { Hono } from 'hono';
import { zValidator } from '@hono/zod-validator';
import { jobManager } from '../services/queue';
import { storage } from '../services/storage';
import { apiLogger as logger } from '../lib/logger';
import { jobIdSchema } from '../lib/validation';
import { JobNotFoundError, AppError } from '../lib/errors';
import { ERROR_CODES } from '@rebloom/shared';
import mime from 'mime-types';

const download = new Hono();

/**
 * GET /api/download/:id
 * Télécharge l'image traitée d'un job
 */
download.get('/:id', zValidator('param', jobIdSchema), async (c) => {
  const { id } = c.req.valid('param');

  const job = await jobManager.get(id);
  if (!job) {
    throw new JobNotFoundError(id);
  }

  if (job.status !== 'completed' || !job.processedImage) {
    throw new AppError(
      ERROR_CODES.JOB_NOT_FOUND,
      'Image not ready for download',
      400,
      { status: job.status }
    );
  }

  const { processedImage } = job;

  try {
    const buffer = await storage.getFile(processedImage.filename);

    logger.info({ jobId: id, filename: processedImage.filename }, 'File downloaded');

    // Headers pour le téléchargement
    c.header('Content-Type', processedImage.mimeType);
    c.header('Content-Length', buffer.length.toString());
    c.header('Content-Disposition', `attachment; filename="${processedImage.originalName}"`);
    c.header('Cache-Control', 'private, max-age=3600');

    return c.body(buffer);
  } catch (error) {
    logger.error({ error, filename: processedImage.filename }, 'Download failed');
    throw new AppError(ERROR_CODES.INTERNAL_ERROR, 'File not found', 404);
  }
});

/**
 * GET /api/files/*
 * Sert les fichiers statiques (images originales et traitées)
 * Utilisé en développement, en prod utiliser un CDN
 */
download.get('/files/*', async (c) => {
  const filePath = c.req.path.replace('/api/files/', '');

  if (!filePath || filePath.includes('..')) {
    throw new AppError(ERROR_CODES.INTERNAL_ERROR, 'Invalid file path', 400);
  }

  try {
    const exists = await storage.exists(filePath);
    if (!exists) {
      throw new AppError(ERROR_CODES.INTERNAL_ERROR, 'File not found', 404);
    }

    const buffer = await storage.getFile(filePath);
    const mimeType = mime.lookup(filePath) || 'application/octet-stream';

    c.header('Content-Type', mimeType);
    c.header('Content-Length', buffer.length.toString());
    c.header('Cache-Control', 'public, max-age=31536000'); // 1 year cache

    return c.body(buffer);
  } catch (error) {
    if (error instanceof AppError) throw error;
    logger.error({ error, filePath }, 'File serving failed');
    throw new AppError(ERROR_CODES.INTERNAL_ERROR, 'File not found', 404);
  }
});

export { download };
