/**
 * ReBloom Shared Types
 * Types partagés entre le frontend et le backend
 */

// ============================================
// Job Status & Types
// ============================================

export type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface Job {
  id: string;
  status: JobStatus;
  progress: number; // 0-100
  createdAt: string;
  updatedAt: string;
  completedAt?: string;

  // Input
  originalImage: ImageInfo;

  // Output (available when completed)
  processedImage?: ImageInfo;

  // Error info (when failed)
  error?: JobError;

  // Processing metadata
  metadata: JobMetadata;
}

export interface ImageInfo {
  id: string;
  filename: string;
  originalName: string;
  mimeType: ImageMimeType;
  size: number; // bytes
  width: number;
  height: number;
  url: string;
}

export interface JobError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface JobMetadata {
  modelUsed: string;
  processingTimeMs?: number;
  scaleFactor?: number;
  provider: AIProvider;
}

// ============================================
// AI Provider Types
// ============================================

export type AIProvider = 'replicate' | 'huggingface' | 'custom';

export interface AIProviderConfig {
  provider: AIProvider;
  modelId: string;
  apiKey?: string;
  baseUrl?: string;
}

export interface EnhanceOptions {
  scaleFactor?: 2 | 4; // Upscaling factor
  faceEnhance?: boolean; // Enable face enhancement
  outputFormat?: OutputFormat;
}

// ============================================
// API Request/Response Types
// ============================================

export interface UploadResponse {
  success: true;
  jobId: string;
  originalImage: ImageInfo;
  estimatedTimeSeconds: number;
}

export interface JobStatusResponse {
  success: true;
  job: Job;
}

export interface DownloadResponse {
  success: true;
  downloadUrl: string;
  expiresAt: string;
}

export interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

export type ApiResponse<T> = T | ErrorResponse;

// ============================================
// Upload Types
// ============================================

export type ImageMimeType = 'image/jpeg' | 'image/png' | 'image/webp';
export type OutputFormat = 'jpeg' | 'png' | 'webp';

export interface UploadValidation {
  maxSizeBytes: number;
  allowedMimeTypes: ImageMimeType[];
  maxDimension: number;
  minDimension: number;
}

// ============================================
// Webhook Types (Replicate)
// ============================================

export interface ReplicateWebhookPayload {
  id: string;
  status: 'starting' | 'processing' | 'succeeded' | 'failed' | 'canceled';
  output?: string | string[];
  error?: string;
  metrics?: {
    predict_time?: number;
  };
}

// ============================================
// Queue Types
// ============================================

export interface ProcessingJobData {
  jobId: string;
  imageId: string;
  imagePath: string;
  options: EnhanceOptions;
  webhookUrl?: string;
}

export interface ProcessingJobResult {
  success: boolean;
  outputPath?: string;
  processingTimeMs: number;
  error?: string;
}
