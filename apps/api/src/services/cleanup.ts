/**
 * Periodic Cleanup Service
 * Removes orphaned files and expired jobs
 */

import { storageLogger as logger } from '../lib/logger';
import { storage } from './storage';
import { jobManager } from './queue';
import { JOB_CONFIG } from '@rebloom/shared';

// ============================================
// Configuration
// ============================================

const CLEANUP_INTERVAL_MS = 60 * 60 * 1000; // Run every hour
const FILE_MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 hours for completed jobs
const FAILED_FILE_MAX_AGE_MS = 2 * 60 * 60 * 1000; // 2 hours for failed job files

// ============================================
// Cleanup Service
// ============================================

class CleanupService {
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private isRunning = false;

  /**
   * Start the periodic cleanup
   */
  start(): void {
    if (this.intervalId) {
      logger.warn('Cleanup service already running');
      return;
    }

    logger.info('Starting cleanup service');

    // Run initial cleanup after 5 minutes
    setTimeout(() => this.runCleanup(), 5 * 60 * 1000);

    // Then run periodically
    this.intervalId = setInterval(() => {
      this.runCleanup();
    }, CLEANUP_INTERVAL_MS);

    // Don't prevent process exit
    this.intervalId.unref();
  }

  /**
   * Stop the cleanup service
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
      logger.info('Cleanup service stopped');
    }
  }

  /**
   * Run cleanup operations
   */
  async runCleanup(): Promise<void> {
    if (this.isRunning) {
      logger.debug('Cleanup already in progress, skipping');
      return;
    }

    this.isRunning = true;
    const startTime = Date.now();

    try {
      logger.info('Starting cleanup run');

      // Cleanup old completed job files
      const completedDeleted = await this.cleanupOldJobFiles('completed', FILE_MAX_AGE_MS);

      // Cleanup old failed job files (faster cleanup)
      const failedDeleted = await this.cleanupOldJobFiles('failed', FAILED_FILE_MAX_AGE_MS);

      const duration = Date.now() - startTime;
      logger.info(
        { completedDeleted, failedDeleted, durationMs: duration },
        'Cleanup run completed'
      );
    } catch (error) {
      logger.error({ error }, 'Cleanup run failed');
    } finally {
      this.isRunning = false;
    }
  }

  /**
   * Cleanup files for jobs with a specific status older than maxAge
   */
  private async cleanupOldJobFiles(status: 'completed' | 'failed', maxAgeMs: number): Promise<number> {
    let deletedCount = 0;
    const now = Date.now();

    try {
      const jobs = await jobManager.list();
      const oldJobs = jobs.filter((job) => {
        if (job.status !== status) return false;
        const jobAge = now - new Date(job.updatedAt).getTime();
        return jobAge > maxAgeMs;
      });

      for (const job of oldJobs) {
        try {
          // Delete original image
          if (job.originalImage?.filename) {
            await storage.delete(job.originalImage.filename);
            deletedCount++;
          }

          // Delete processed image (for completed jobs)
          if (job.processedImage?.filename) {
            await storage.delete(job.processedImage.filename);
            deletedCount++;
          }

          // Delete the job record
          await jobManager.delete(job.id);

          logger.debug({ jobId: job.id, status }, 'Cleaned up old job');
        } catch (error) {
          logger.warn({ jobId: job.id, error }, 'Failed to cleanup job');
        }
      }
    } catch (error) {
      logger.error({ error, status }, 'Failed to list jobs for cleanup');
    }

    return deletedCount;
  }
}

export const cleanupService = new CleanupService();
