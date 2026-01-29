/**
 * Route d'upload d'images
 * POST /api/upload
 *
 * Sécurisé avec:
 * - Rate limiting strict
 * - Validation de type MIME (magic bytes)
 * - Sanitization des noms de fichiers
 * - Validation de taille
 * - Validation des dimensions d'image
 */

import { Hono } from 'hono';
import { nanoid } from 'nanoid';
import sharp from 'sharp';
import { storage } from '../services/storage';
import { jobManager } from '../services/queue';
import { apiLogger as logger } from '../lib/logger';
import { uploadOptionsSchema, validateImageMimeType, validateImageSize, validateImageMagicBytes, validateImageDimensions, sanitizeFilename } from '../lib/validation';
import { FileTooLargeError, InvalidFileTypeError, ValidationError, ImageDimensionError } from '../lib/errors';
import { UPLOAD_CONFIG, DEFAULT_MODEL, JOB_CONFIG } from '@rebloom/shared';
import { uploadRateLimit } from '../lib/rate-limit';
import { requireAuth, checkQuota, recordUsage } from '../lib/auth';
import type { ImageMimeType, UploadResponse, EnhanceOptions } from '@rebloom/shared';

const upload = new Hono();

// Apply authentication, quota check, and rate limiting to uploads
upload.use('/', requireAuth());
upload.use('/', checkQuota());
upload.use('/', uploadRateLimit);

upload.post('/', async (c) => {
  const startTime = Date.now();
  const requestId = c.get('requestId') || 'unknown';
  const userId = c.get('userId');

  try {
    // Parser le body multipart
    const body = await c.req.parseBody();
    const file = body['file'];
    const options = body['options'] as string | undefined;

    // Valider que le fichier existe
    if (!file || !(file instanceof File)) {
      throw new ValidationError('No file provided. Please upload an image file.');
    }

    // Sanitize filename for logging (security)
    const safeFilename = sanitizeFilename(file.name);
    
    logger.info(
      {
        requestId,
        filename: safeFilename,
        size: file.size,
        type: file.type,
      },
      'Upload received'
    );

    // Valider la taille avant de lire le buffer (fail fast)
    if (!validateImageSize(file.size)) {
      throw new FileTooLargeError(UPLOAD_CONFIG.maxSizeBytes, file.size);
    }

    // Convertir en buffer pour validation des magic bytes
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    // Valider le type réel du fichier via magic bytes (sécurité)
    const magicBytesResult = validateImageMagicBytes(buffer);
    if (!magicBytesResult.valid) {
      logger.warn(
        { requestId, claimedType: file.type, detectedType: magicBytesResult.detectedType },
        'Invalid file type detected via magic bytes'
      );
      throw new InvalidFileTypeError(
        magicBytesResult.detectedType || file.type, 
        [...UPLOAD_CONFIG.allowedMimeTypes]
      );
    }

    // Utiliser le type détecté (plus sûr que le type déclaré)
    const actualMimeType = magicBytesResult.detectedType as ImageMimeType;

    // Valider les dimensions de l'image avec Sharp
    const metadata = await sharp(buffer).metadata();
    if (!metadata.width || !metadata.height) {
      throw new ValidationError('Could not read image dimensions. The file may be corrupted.');
    }

    if (!validateImageDimensions(metadata.width, metadata.height)) {
      logger.warn(
        { requestId, width: metadata.width, height: metadata.height },
        'Invalid image dimensions'
      );
      throw new ImageDimensionError(
        metadata.width,
        metadata.height,
        UPLOAD_CONFIG.minDimension,
        UPLOAD_CONFIG.maxDimension
      );
    }

    logger.debug(
      { requestId, width: metadata.width, height: metadata.height },
      'Image dimensions validated'
    );

    // Parser les options
    let enhanceOptions: EnhanceOptions = {
      scaleFactor: 4,
      faceEnhance: false,
      outputFormat: 'png',
    };

    if (options) {
      try {
        const parsed = JSON.parse(options);
        const validated = uploadOptionsSchema.parse(parsed);
        enhanceOptions = {
          scaleFactor: Number(validated.scaleFactor) as 2 | 4,
          faceEnhance: validated.faceEnhance,
          outputFormat: validated.outputFormat,
        };
      } catch (e) {
        logger.warn({ requestId, error: e }, 'Invalid options, using defaults');
      }
    }

    // Sauvegarder l'image originale avec le nom sanitizé et le type réel
    const imageInfo = await storage.saveOriginal(
      buffer,
      safeFilename,
      actualMimeType
    );

    // Créer le job avec l'userId
    const jobId = nanoid(16);
    const job = await jobManager.create({
      id: jobId,
      userId, // Associate job with user
      originalImage: imageInfo,
      metadata: {
        modelUsed: DEFAULT_MODEL.name,
        provider: 'replicate',
        scaleFactor: enhanceOptions.scaleFactor,
      },
    });

    // Ajouter à la queue
    await jobManager.enqueue(jobId, {
      jobId,
      imageId: imageInfo.id,
      imagePath: imageInfo.filename,
      options: enhanceOptions,
    });

    const processingTime = Date.now() - startTime;
    logger.info({ jobId, userId, processingTime }, 'Upload processed successfully');

    // Record usage for quota tracking
    if (userId) {
      await recordUsage(userId);
    }

    // Estimer le temps de traitement (basé sur la taille de l'image)
    const estimatedSeconds = Math.max(10, Math.ceil(imageInfo.width * imageInfo.height / 500000));

    const response: UploadResponse = {
      success: true,
      jobId: job.id,
      originalImage: imageInfo,
      estimatedTimeSeconds: estimatedSeconds,
    };

    return c.json(response, 201);
  } catch (error) {
    logger.error({ error }, 'Upload failed');
    throw error;
  }
});

export { upload };
