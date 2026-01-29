/**
 * ReBloom Shared Constants
 * Configuration partagée entre le frontend et le backend
 */

import type { ImageMimeType, UploadValidation, OutputFormat } from './types';

// ============================================
// Upload Configuration
// ============================================

export const UPLOAD_CONFIG: UploadValidation = {
  maxSizeBytes: 20 * 1024 * 1024, // 20MB
  allowedMimeTypes: ['image/jpeg', 'image/png', 'image/webp'],
  maxDimension: 4096, // Max 4K
  minDimension: 64, // Min 64px
};

export const MIME_TYPE_EXTENSIONS: Record<ImageMimeType, string> = {
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
};

export const OUTPUT_FORMATS: OutputFormat[] = ['jpeg', 'png', 'webp'];

// ============================================
// Job Configuration
// ============================================

export const JOB_CONFIG = {
  /** Time before a job is considered stale */
  staleAfterMs: 24 * 60 * 60 * 1000, // 24 hours

  /** Maximum processing time before timeout */
  timeoutMs: 5 * 60 * 1000, // 5 minutes

  /** Number of retry attempts */
  maxRetries: 3,

  /** Polling interval for status updates */
  pollingIntervalMs: 2000, // 2 seconds

  /** Progress update granularity */
  progressSteps: {
    uploaded: 10,
    queued: 20,
    processing: 50,
    enhancing: 80,
    saving: 95,
    completed: 100,
  },
} as const;

// ============================================
// AI Models
// ============================================

export const AI_MODELS = {
  replicate: {
    // Real-ESRGAN - Best for general image enhancement
    realEsrgan: {
      id: 'nightmareai/real-esrgan:f121d640bd286e1fdc67f9799164c1d5be36ff74576ee11c803ae5b665dd46aa',
      name: 'Real-ESRGAN',
      description: 'Excellent pour le défloutage général et upscaling',
      supportsScale: [2, 4],
      supportsFaceEnhance: true,
    },
    // SwinIR - Good for restoration
    swinir: {
      id: 'jingyunliang/swinir:660d922d33153019e8c263a3bba265de882e7f4f70396571c8c4b03e0c40c6bb',
      name: 'SwinIR',
      description: 'Performant pour la restauration de détails',
      supportsScale: [2, 4],
      supportsFaceEnhance: false,
    },
  },
} as const;

export const DEFAULT_MODEL = AI_MODELS.replicate.realEsrgan;

// ============================================
// API Routes
// ============================================

export const API_ROUTES = {
  upload: '/api/upload',
  jobs: '/api/jobs',
  jobStatus: (id: string) => `/api/jobs/${id}`,
  jobResult: (id: string) => `/api/jobs/${id}/result`,
  download: (id: string) => `/api/download/${id}`,
  webhook: '/api/webhooks/replicate',
} as const;

// ============================================
// Error Codes
// ============================================

export const ERROR_CODES = {
  // Upload errors
  FILE_TOO_LARGE: 'FILE_TOO_LARGE',
  INVALID_FILE_TYPE: 'INVALID_FILE_TYPE',
  IMAGE_TOO_SMALL: 'IMAGE_TOO_SMALL',
  IMAGE_TOO_LARGE: 'IMAGE_TOO_LARGE',
  UPLOAD_FAILED: 'UPLOAD_FAILED',

  // Job errors
  JOB_NOT_FOUND: 'JOB_NOT_FOUND',
  JOB_FAILED: 'JOB_FAILED',
  JOB_TIMEOUT: 'JOB_TIMEOUT',
  JOB_ALREADY_PROCESSED: 'JOB_ALREADY_PROCESSED',

  // Processing errors
  PROCESSING_FAILED: 'PROCESSING_FAILED',
  MODEL_ERROR: 'MODEL_ERROR',
  PROVIDER_ERROR: 'PROVIDER_ERROR',

  // General errors
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  RATE_LIMIT_EXCEEDED: 'RATE_LIMIT_EXCEEDED',
  UNAUTHORIZED: 'UNAUTHORIZED',
} as const;

export type ErrorCode = (typeof ERROR_CODES)[keyof typeof ERROR_CODES];

// ============================================
// UI Configuration
// ============================================

export const UI_CONFIG = {
  /** Max images shown in history */
  maxHistoryItems: 10,

  /** Animation durations (ms) */
  animations: {
    fadeIn: 200,
    slideIn: 300,
    progress: 100,
  },

  /** Comparison slider default position */
  sliderDefaultPosition: 50,
} as const;
