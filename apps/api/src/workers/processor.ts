/**
 * Worker de traitement des images
 * Consomme les jobs de la queue et appelle le provider AI
 * Includes automatic cleanup of orphaned files on failure
 */

import { Worker, Job as BullJob } from 'bullmq';
import { Redis } from 'ioredis';
import { config_ } from '../lib/config';
import { jobLogger as logger } from '../lib/logger';
import { jobManager } from '../services/queue';
import { storage } from '../services/storage';
import { getAIProvider } from '../services/ai/provider';
import { JOB_CONFIG, ERROR_CODES } from '@rebloom/shared';
import type { ProcessingJobData, ProcessingJobResult, OutputFormat } from '@rebloom/shared';

// ============================================
// File Cleanup Helper
// ============================================

/**
 * Cleanup orphaned files when processing fails permanently
 * Only cleans up on final attempt to allow retries
 */
async function cleanupOrphanedFiles(
  imagePath: string,
  jobId: string,
  attemptsMade: number,
  maxAttempts: number
): Promise<void> {
  // Only cleanup on final attempt (no more retries)
  if (attemptsMade < maxAttempts) {
    logger.debug(
      { jobId, attemptsMade, maxAttempts },
      'Skipping cleanup - retries remaining'
    );
    return;
  }

  logger.info({ jobId, imagePath }, 'Cleaning up orphaned files after final failure');

  try {
    // Delete original image
    await storage.delete(imagePath);
    logger.debug({ jobId, path: imagePath }, 'Deleted original image');
  } catch (error) {
    logger.warn(
      { jobId, path: imagePath, error },
      'Failed to delete original image during cleanup'
    );
  }

  // Also try to delete any partial processed output
  const possibleOutputExtensions = ['.png', '.jpg', '.jpeg', '.webp'];
  for (const ext of possibleOutputExtensions) {
    try {
      const processedPath = `processed/${jobId}${ext}`;
      if (await storage.exists(processedPath)) {
        await storage.delete(processedPath);
        logger.debug({ jobId, path: processedPath }, 'Deleted partial processed image');
      }
    } catch {
      // Ignore errors for non-existent files
    }
  }
}

// ============================================
// Redis Connection
// ============================================

const connection = new Redis(config_.redis.url, {
  maxRetriesPerRequest: null,
});

// ============================================
// Worker Definition
// ============================================

const QUEUE_NAME = 'image-processing';

async function processJob(
  bullJob: BullJob<ProcessingJobData>
): Promise<ProcessingJobResult> {
  const { jobId, imagePath, options } = bullJob.data;
  const startTime = Date.now();

  logger.info({ jobId, imagePath }, 'Processing started');

  try {
    // Mettre à jour le statut
    await jobManager.updateStatus(jobId, 'processing', JOB_CONFIG.progressSteps.processing);

    // Récupérer le provider AI
    const provider = getAIProvider();

    // Construire l'URL de l'image
    // En local, on doit exposer l'image publiquement pour Replicate
    // Option 1: Utiliser ngrok ou similar
    // Option 2: Upload temporaire vers un service public
    // Option 3: Lire le fichier et l'envoyer en base64 (si supporté)

    // Pour le dev, on utilise le chemin local et on lit le fichier
    const imageBuffer = await storage.getFile(imagePath);
    const base64Image = `data:image/png;base64,${imageBuffer.toString('base64')}`;

    // Mettre à jour progress
    await jobManager.updateStatus(jobId, 'processing', JOB_CONFIG.progressSteps.enhancing);

    // Appeler le provider
    const result = await provider.enhance(base64Image, options);

    if (!result.success || !result.outputUrl) {
      throw new Error(result.error || 'Enhancement failed without error message');
    }

    // Télécharger l'image résultante
    await jobManager.updateStatus(jobId, 'processing', JOB_CONFIG.progressSteps.saving);

    const outputResponse = await fetch(result.outputUrl);
    if (!outputResponse.ok) {
      throw new Error(`Failed to download processed image: ${outputResponse.status}`);
    }

    const outputBuffer = Buffer.from(await outputResponse.arrayBuffer());

    // Sauvegarder l'image traitée
    const processedImage = await storage.saveProcessed(
      outputBuffer,
      jobId,
      options.outputFormat as OutputFormat
    );

    // Mettre à jour le job comme terminé
    await jobManager.setCompleted(jobId, processedImage);

    const processingTimeMs = Date.now() - startTime;

    logger.info(
      { jobId, processingTimeMs, outputSize: outputBuffer.length },
      'Processing completed'
    );

    return {
      success: true,
      outputPath: processedImage.filename,
      processingTimeMs,
    };
  } catch (error) {
    const processingTimeMs = Date.now() - startTime;
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';

    logger.error({ jobId, error: errorMessage, processingTimeMs }, 'Processing failed');

    // Marquer le job comme échoué
    await jobManager.setFailed(jobId, {
      code: ERROR_CODES.PROCESSING_FAILED,
      message: errorMessage,
    });

    // Cleanup orphaned files on final failure
    const attemptsMade = bullJob.attemptsMade + 1; // Current attempt
    const maxAttempts = bullJob.opts.attempts || config_.jobs.maxRetries;
    await cleanupOrphanedFiles(imagePath, jobId, attemptsMade, maxAttempts);

    return {
      success: false,
      error: errorMessage,
      processingTimeMs,
    };
  }
}

// ============================================
// Worker Instance
// ============================================

export const worker = new Worker<ProcessingJobData, ProcessingJobResult>(
  QUEUE_NAME,
  processJob,
  {
    connection,
    concurrency: 2, // Traiter 2 jobs en parallèle max
    limiter: {
      max: 10,
      duration: 60000, // Max 10 jobs par minute (rate limiting Replicate)
    },
  }
);

// ============================================
// Worker Events
// ============================================

worker.on('ready', () => {
  logger.info('Worker ready and listening for jobs');
});

worker.on('completed', (job) => {
  logger.info({ jobId: job.id }, 'Worker completed job');
});

worker.on('failed', (job, error) => {
  logger.error({ jobId: job?.id, error: error.message }, 'Worker job failed');
});

worker.on('error', (error) => {
  logger.error({ error: error.message }, 'Worker error');
});

// ============================================
// Graceful Shutdown
// ============================================

async function shutdown() {
  logger.info('Shutting down worker...');
  await worker.close();
  await connection.quit();
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

// Si exécuté directement (pas importé)
if (import.meta.url === `file://${process.argv[1]}`) {
  logger.info('Worker started in standalone mode');
}
