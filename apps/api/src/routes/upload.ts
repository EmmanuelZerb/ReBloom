/**
 * Route d'upload d'images
 * POST /api/upload
 * 
 * Sécurisé avec:
 * - Rate limiting strict
 * - Validation de type MIME (magic bytes)
 * - Sanitization des noms de fichiers
 * - Validation de taille
 */

import { Hono } from 'hono';
import { nanoid } from 'nanoid';
import { storage } from '../services/storage';
import { jobManager } from '../services/queue';
import { apiLogger as logger } from '../lib/logger';
import { uploadOptionsSchema, validateImageMimeType, validateImageSize, validateImageMagicBytes, sanitizeFilename } from '../lib/validation';
import { FileTooLargeError, InvalidFileTypeError, ValidationError } from '../lib/errors';
import { UPLOAD_CONFIG, DEFAULT_MODEL, JOB_CONFIG } from '@rebloom/shared';
import { uploadRateLimit } from '../lib/rate-limit';
import type { ImageMimeType, UploadResponse, EnhanceOptions } from '@rebloom/shared';

const upload = new Hono();

// Apply strict rate limiting to uploads
upload.use('/', uploadRateLimit);

upload.post('/', async (c) => {
  const startTime = Date.now();
  const requestId = c.get('requestId') || 'unknown';

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

    // Créer le job
    const jobId = nanoid(16);
    const job = await jobManager.create({
      id: jobId,
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
    logger.info({ jobId, processingTime }, 'Upload processed successfully');

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
