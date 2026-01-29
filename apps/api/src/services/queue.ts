/**
 * Service de queue avec BullMQ
 * Gestion des jobs de traitement asynchrones
 */

import { Queue, Worker, Job as BullJob } from 'bullmq';
import { Redis } from 'ioredis';
import { config_ } from '../lib/config';
import { jobLogger as logger } from '../lib/logger';
import type { Job, JobStatus, ProcessingJobData, ProcessingJobResult } from '@rebloom/shared';
import { JOB_CONFIG } from '@rebloom/shared';

// ============================================
// Redis Connection
// ============================================

const connection = new Redis(config_.redis.url, {
  maxRetriesPerRequest: null, // Required for BullMQ
});

connection.on('connect', () => logger.info('Redis connected'));
connection.on('error', (err) => logger.error({ err }, 'Redis error'));

// ============================================
// Queue Definition
// ============================================

const QUEUE_NAME = 'image-processing';

export const processingQueue = new Queue<ProcessingJobData, ProcessingJobResult>(QUEUE_NAME, {
  connection,
  defaultJobOptions: {
    attempts: config_.jobs.maxRetries,
    backoff: {
      type: 'exponential',
      delay: 2000,
    },
    removeOnComplete: {
      age: 24 * 3600, // Keep completed jobs for 24h
      count: 1000,
    },
    removeOnFail: {
      age: 7 * 24 * 3600, // Keep failed jobs for 7 days
    },
  },
});

// ============================================
// In-Memory Job Store (simple implementation)
// En production, utiliser Redis ou une DB
// ============================================

const jobStore = new Map<string, Job>();

export const jobManager = {
  /**
   * Crée un nouveau job
   */
  async create(jobData: Omit<Job, 'status' | 'progress' | 'createdAt' | 'updatedAt'>): Promise<Job> {
    const now = new Date().toISOString();
    const job: Job = {
      ...jobData,
      status: 'pending',
      progress: JOB_CONFIG.progressSteps.uploaded,
      createdAt: now,
      updatedAt: now,
    };

    jobStore.set(job.id, job);
    logger.info({ jobId: job.id }, 'Job created');
    return job;
  },

  /**
   * Récupère un job par ID
   */
  async get(jobId: string): Promise<Job | null> {
    return jobStore.get(jobId) || null;
  },

  /**
   * Met à jour le statut d'un job
   */
  async updateStatus(
    jobId: string,
    status: JobStatus,
    progress?: number,
    extra?: Partial<Job>
  ): Promise<Job | null> {
    const job = jobStore.get(jobId);
    if (!job) return null;

    const updated: Job = {
      ...job,
      ...extra,
      status,
      progress: progress ?? job.progress,
      updatedAt: new Date().toISOString(),
      ...(status === 'completed' && { completedAt: new Date().toISOString() }),
    };

    jobStore.set(jobId, updated);
    logger.info({ jobId, status, progress }, 'Job status updated');
    return updated;
  },

  /**
   * Marque un job comme échoué
   */
  async setFailed(jobId: string, error: { code: string; message: string }): Promise<Job | null> {
    return this.updateStatus(jobId, 'failed', undefined, { error });
  },

  /**
   * Marque un job comme terminé
   */
  async setCompleted(jobId: string, processedImage: Job['processedImage']): Promise<Job | null> {
    return this.updateStatus(jobId, 'completed', JOB_CONFIG.progressSteps.completed, {
      processedImage,
    });
  },

  /**
   * Ajoute un job à la queue de traitement
   */
  async enqueue(jobId: string, data: ProcessingJobData): Promise<void> {
    await processingQueue.add(jobId, data, { jobId });
    await this.updateStatus(jobId, 'pending', JOB_CONFIG.progressSteps.queued);
    logger.info({ jobId }, 'Job enqueued');
  },

  /**
   * Supprime un job
   */
  async delete(jobId: string): Promise<void> {
    jobStore.delete(jobId);
    logger.info({ jobId }, 'Job deleted');
  },

  /**
   * Liste tous les jobs (pour debug)
   */
  async list(): Promise<Job[]> {
    return Array.from(jobStore.values());
  },
};

// ============================================
// Queue Events (pour monitoring)
// ============================================

processingQueue.on('waiting', ({ jobId }) => {
  logger.debug({ jobId }, 'Job waiting');
});

processingQueue.on('active', ({ jobId }) => {
  logger.debug({ jobId }, 'Job active');
});

processingQueue.on('completed', ({ jobId }) => {
  logger.debug({ jobId }, 'Job completed');
});

processingQueue.on('failed', ({ jobId, failedReason }) => {
  logger.error({ jobId, failedReason }, 'Job failed');
});

// ============================================
// Cleanup
// ============================================

export async function closeQueue(): Promise<void> {
  await processingQueue.close();
  await connection.quit();
  logger.info('Queue closed');
}
