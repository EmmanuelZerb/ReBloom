/**
 * Schémas de validation Zod pour l'API
 */

import { z } from 'zod';
import { UPLOAD_CONFIG } from '@rebloom/shared';

// ============================================
// Upload Validation
// ============================================

export const uploadOptionsSchema = z.object({
  scaleFactor: z.enum(['2', '4']).optional().default('4'),
  faceEnhance: z
    .enum(['true', 'false'])
    .optional()
    .default('false')
    .transform((v) => v === 'true'),
  outputFormat: z.enum(['jpeg', 'png', 'webp']).optional().default('png'),
});

export type UploadOptions = z.infer<typeof uploadOptionsSchema>;

// ============================================
// Job ID Validation
// ============================================

export const jobIdSchema = z.object({
  id: z.string().min(1, 'Job ID is required'),
});

export type JobIdParams = z.infer<typeof jobIdSchema>;

// ============================================
// Webhook Validation (Replicate)
// ============================================

export const replicateWebhookSchema = z.object({
  id: z.string(),
  status: z.enum(['starting', 'processing', 'succeeded', 'failed', 'canceled']),
  output: z.union([z.string(), z.array(z.string())]).optional(),
  error: z.string().optional(),
  metrics: z
    .object({
      predict_time: z.number().optional(),
    })
    .optional(),
});

export type ReplicateWebhook = z.infer<typeof replicateWebhookSchema>;

// ============================================
// Image Validation Helpers
// ============================================

export function validateImageMimeType(mimeType: string): boolean {
  return UPLOAD_CONFIG.allowedMimeTypes.includes(mimeType as any);
}

export function validateImageSize(sizeBytes: number): boolean {
  return sizeBytes <= UPLOAD_CONFIG.maxSizeBytes;
}

export function validateImageDimensions(width: number, height: number): boolean {
  const { minDimension, maxDimension } = UPLOAD_CONFIG;
  return (
    width >= minDimension &&
    width <= maxDimension &&
    height >= minDimension &&
    height <= maxDimension
  );
}

// ============================================
// Security: Magic Bytes Validation
// ============================================

/**
 * Validates file type by checking magic bytes (file signature)
 * This prevents MIME type spoofing attacks
 */
export function validateImageMagicBytes(buffer: Buffer): { valid: boolean; detectedType: string | null } {
  // Magic bytes signatures for common image formats
  const signatures: Record<string, { bytes: number[]; offset?: number }> = {
    'image/jpeg': { bytes: [0xFF, 0xD8, 0xFF] },
    'image/png': { bytes: [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] },
    'image/webp': { bytes: [0x52, 0x49, 0x46, 0x46], offset: 0 }, // RIFF header
    'image/gif': { bytes: [0x47, 0x49, 0x46, 0x38] }, // GIF87a or GIF89a
  };
  
  // Additional check for WebP (WEBP marker at offset 8)
  const webpMarker = [0x57, 0x45, 0x42, 0x50];
  
  for (const [mimeType, signature] of Object.entries(signatures)) {
    const offset = signature.offset ?? 0;
    let matches = true;
    
    for (let i = 0; i < signature.bytes.length; i++) {
      if (buffer[offset + i] !== signature.bytes[i]) {
        matches = false;
        break;
      }
    }
    
    if (matches) {
      // Special handling for WebP: verify WEBP marker
      if (mimeType === 'image/webp') {
        let hasWebpMarker = true;
        for (let i = 0; i < webpMarker.length; i++) {
          if (buffer[8 + i] !== webpMarker[i]) {
            hasWebpMarker = false;
            break;
          }
        }
        if (!hasWebpMarker) continue;
      }
      
      const isAllowed = UPLOAD_CONFIG.allowedMimeTypes.includes(mimeType as any);
      return { valid: isAllowed, detectedType: mimeType };
    }
  }
  
  return { valid: false, detectedType: null };
}

// ============================================
// Security: Filename Sanitization
// ============================================

/**
 * Sanitizes filenames to prevent path traversal and other attacks
 */
export function sanitizeFilename(filename: string): string {
  // Remove path components
  let sanitized = filename.replace(/^.*[\\\/]/, '');
  
  // Remove null bytes
  sanitized = sanitized.replace(/\x00/g, '');
  
  // Remove control characters
  sanitized = sanitized.replace(/[\x00-\x1f\x7f]/g, '');
  
  // Remove problematic characters
  sanitized = sanitized.replace(/[<>:"/\\|?*]/g, '_');
  
  // Remove leading dots (hidden files)
  sanitized = sanitized.replace(/^\.+/, '');
  
  // Limit length
  if (sanitized.length > 200) {
    const ext = sanitized.slice(sanitized.lastIndexOf('.'));
    sanitized = sanitized.slice(0, 200 - ext.length) + ext;
  }
  
  // Ensure filename is not empty
  if (!sanitized || sanitized === '') {
    sanitized = 'upload';
  }
  
  return sanitized;
}

/**
 * Validates and sanitizes a complete upload request
 */
export interface SanitizedUpload {
  filename: string;
  mimeType: string;
  buffer: Buffer;
}

export function validateAndSanitizeUpload(
  file: File,
  buffer: Buffer
): SanitizedUpload {
  // Sanitize filename
  const filename = sanitizeFilename(file.name);
  
  // Validate magic bytes (actual file content)
  const { valid, detectedType } = validateImageMagicBytes(buffer);
  
  if (!valid || !detectedType) {
    throw new Error('Invalid or unsupported image format detected');
  }
  
  // Ensure claimed MIME type matches detected type
  if (file.type !== detectedType && file.type !== 'application/octet-stream') {
    // Log mismatch but use detected type (more secure)
    console.warn(`MIME type mismatch: claimed ${file.type}, detected ${detectedType}`);
  }
  
  return {
    filename,
    mimeType: detectedType,
    buffer,
  };
}
