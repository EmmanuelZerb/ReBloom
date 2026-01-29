/**
 * Service de queue avec BullMQ
 * Gestion des jobs de traitement asynchrones
 * Persistance des jobs dans Redis
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

// Separate connection for job store operations
const jobStoreRedis = new Redis(config_.redis.url, {
  maxRetriesPerRequest: 3,
  retryDelayOnFailover: 100,
});

connection.on('connect', () => logger.info('Redis (BullMQ) connected'));
connection.on('error', (err) => logger.error({ err }, 'Redis (BullMQ) error'));

jobStoreRedis.on('connect', () => logger.info('Redis (JobStore) connected'));
jobStoreRedis.on('error', (err) => logger.error({ err }, 'Redis (JobStore) error'));

// ============================================
// Queue Definition
// ============================================

const QUEUE_NAME = 'image-processing';
const JOB_STORE_PREFIX = 'rebloom:job:';
const JOB_TTL_SECONDS = 7 * 24 * 60 * 60; // 7 days

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
// Redis Job Store (persistent)
// ============================================

export const jobManager = {
  /**
   * Crée un nouveau job (persisté dans Redis)
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

    const key = `${JOB_STORE_PREFIX}${job.id}`;
    await jobStoreRedis.setex(key, JOB_TTL_SECONDS, JSON.stringify(job));
    logger.info({ jobId: job.id }, 'Job created in Redis');
    return job;
  },

  /**
   * Récupère un job par ID depuis Redis
   */
  async get(jobId: string): Promise<Job | null> {
    const key = `${JOB_STORE_PREFIX}${jobId}`;
    const data = await jobStoreRedis.get(key);
    if (!data) return null;

    try {
      return JSON.parse(data) as Job;
    } catch (error) {
      logger.error({ jobId, error }, 'Failed to parse job from Redis');
      return null;
    }
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
    const job = await this.get(jobId);
    if (!job) return null;

    const updated: Job = {
      ...job,
      ...extra,
      status,
      progress: progress ?? job.progress,
      updatedAt: new Date().toISOString(),
      ...(status === 'completed' && { completedAt: new Date().toISOString() }),
    };

    const key = `${JOB_STORE_PREFIX}${jobId}`;
    // Refresh TTL on update
    await jobStoreRedis.setex(key, JOB_TTL_SECONDS, JSON.stringify(updated));
    logger.info({ jobId, status, progress }, 'Job status updated in Redis');
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
    const key = `${JOB_STORE_PREFIX}${jobId}`;
    await jobStoreRedis.del(key);
    logger.info({ jobId }, 'Job deleted from Redis');
  },

  /**
   * Liste tous les jobs (pour debug/admin)
   */
  async list(): Promise<Job[]> {
    const keys = await jobStoreRedis.keys(`${JOB_STORE_PREFIX}*`);
    if (keys.length === 0) return [];

    const jobs: Job[] = [];
    for (const key of keys) {
      const data = await jobStoreRedis.get(key);
      if (data) {
        try {
          jobs.push(JSON.parse(data) as Job);
        } catch {
          // Skip invalid entries
        }
      }
    }
    return jobs.sort((a, b) =>
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
  },

  /**
   * Compte le nombre de jobs par statut
   */
  async countByStatus(): Promise<Record<JobStatus, number>> {
    const jobs = await this.list();
    return jobs.reduce(
      (acc, job) => {
        acc[job.status] = (acc[job.status] || 0) + 1;
        return acc;
      },
      { pending: 0, processing: 0, completed: 0, failed: 0 } as Record<JobStatus, number>
    );
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
