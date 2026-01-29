/**
 * Tests for validation utilities
 */

import { describe, it, expect } from 'vitest';
import {
  validateImageMimeType,
  validateImageSize,
  validateImageDimensions,
  validateImageMagicBytes,
  sanitizeFilename,
} from '../src/lib/validation';
import { UPLOAD_CONFIG } from '@rebloom/shared';

describe('Validation Utils', () => {
  describe('validateImageMimeType', () => {
    it('should accept valid MIME types', () => {
      expect(validateImageMimeType('image/jpeg')).toBe(true);
      expect(validateImageMimeType('image/png')).toBe(true);
      expect(validateImageMimeType('image/webp')).toBe(true);
    });

    it('should reject invalid MIME types', () => {
      expect(validateImageMimeType('image/gif')).toBe(false);
      expect(validateImageMimeType('image/bmp')).toBe(false);
      expect(validateImageMimeType('application/pdf')).toBe(false);
      expect(validateImageMimeType('text/plain')).toBe(false);
    });
  });

  describe('validateImageSize', () => {
    it('should accept files within size limit', () => {
      expect(validateImageSize(1024)).toBe(true); // 1KB
      expect(validateImageSize(1024 * 1024)).toBe(true); // 1MB
      expect(validateImageSize(UPLOAD_CONFIG.maxSizeBytes)).toBe(true); // Exactly at limit
    });

    it('should reject files exceeding size limit', () => {
      expect(validateImageSize(UPLOAD_CONFIG.maxSizeBytes + 1)).toBe(false);
      expect(validateImageSize(100 * 1024 * 1024)).toBe(false); // 100MB
    });

    it('should accept zero size', () => {
      expect(validateImageSize(0)).toBe(true);
    });
  });

  describe('validateImageDimensions', () => {
    const { minDimension, maxDimension } = UPLOAD_CONFIG;

    it('should accept valid dimensions', () => {
      expect(validateImageDimensions(500, 500)).toBe(true);
      expect(validateImageDimensions(1920, 1080)).toBe(true);
      expect(validateImageDimensions(minDimension, minDimension)).toBe(true);
      expect(validateImageDimensions(maxDimension, maxDimension)).toBe(true);
    });

    it('should reject dimensions below minimum', () => {
      expect(validateImageDimensions(minDimension - 1, 500)).toBe(false);
      expect(validateImageDimensions(500, minDimension - 1)).toBe(false);
      expect(validateImageDimensions(10, 10)).toBe(false);
    });

    it('should reject dimensions above maximum', () => {
      expect(validateImageDimensions(maxDimension + 1, 500)).toBe(false);
      expect(validateImageDimensions(500, maxDimension + 1)).toBe(false);
      expect(validateImageDimensions(10000, 10000)).toBe(false);
    });
  });

  describe('validateImageMagicBytes', () => {
    it('should detect JPEG files', () => {
      const jpegBuffer = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10]);
      const result = validateImageMagicBytes(jpegBuffer);
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('image/jpeg');
    });

    it('should detect PNG files', () => {
      const pngBuffer = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
      const result = validateImageMagicBytes(pngBuffer);
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('image/png');
    });

    it('should detect WebP files', () => {
      // RIFF....WEBP
      const webpBuffer = Buffer.from([
        0x52, 0x49, 0x46, 0x46, // RIFF
        0x00, 0x00, 0x00, 0x00, // Size (placeholder)
        0x57, 0x45, 0x42, 0x50, // WEBP
      ]);
      const result = validateImageMagicBytes(webpBuffer);
      expect(result.valid).toBe(true);
      expect(result.detectedType).toBe('image/webp');
    });

    it('should reject unknown file types', () => {
      const unknownBuffer = Buffer.from([0x00, 0x01, 0x02, 0x03, 0x04, 0x05]);
      const result = validateImageMagicBytes(unknownBuffer);
      expect(result.valid).toBe(false);
      expect(result.detectedType).toBeNull();
    });

    it('should reject PDF files masquerading as images', () => {
      const pdfBuffer = Buffer.from('%PDF-1.4');
      const result = validateImageMagicBytes(pdfBuffer);
      expect(result.valid).toBe(false);
    });

    it('should reject executable files', () => {
      const exeBuffer = Buffer.from([0x4d, 0x5a]); // MZ header
      const result = validateImageMagicBytes(exeBuffer);
      expect(result.valid).toBe(false);
    });
  });

  describe('sanitizeFilename', () => {
    it('should keep valid filenames unchanged', () => {
      expect(sanitizeFilename('image.jpg')).toBe('image.jpg');
      expect(sanitizeFilename('my-photo.png')).toBe('my-photo.png');
      expect(sanitizeFilename('image_2024.webp')).toBe('image_2024.webp');
    });

    it('should remove path traversal attempts', () => {
      expect(sanitizeFilename('../../../etc/passwd')).toBe('passwd');
      expect(sanitizeFilename('..\\..\\windows\\system32')).toBe('system32');
      expect(sanitizeFilename('/absolute/path/file.jpg')).toBe('file.jpg');
    });

    it('should remove null bytes', () => {
      expect(sanitizeFilename('file\x00.jpg')).toBe('file.jpg');
      expect(sanitizeFilename('image\x00\x00.png')).toBe('image.png');
    });

    it('should remove control characters', () => {
      expect(sanitizeFilename('file\n.jpg')).toBe('file.jpg');
      expect(sanitizeFilename('image\r\n.png')).toBe('image.png');
      expect(sanitizeFilename('photo\t.webp')).toBe('photo.webp');
    });

    it('should replace problematic characters', () => {
      expect(sanitizeFilename('file<name>.jpg')).toBe('file_name_.jpg');
      expect(sanitizeFilename('image:name.png')).toBe('image_name.png');
      expect(sanitizeFilename('photo|test.webp')).toBe('photo_test.webp');
    });

    it('should remove leading dots', () => {
      expect(sanitizeFilename('.hidden')).toBe('hidden');
      expect(sanitizeFilename('..hidden')).toBe('hidden');
      expect(sanitizeFilename('...file.jpg')).toBe('file.jpg');
    });

    it('should truncate long filenames', () => {
      const longName = 'a'.repeat(300) + '.jpg';
      const result = sanitizeFilename(longName);
      expect(result.length).toBeLessThanOrEqual(200);
      expect(result.endsWith('.jpg')).toBe(true);
    });

    it('should handle empty filenames', () => {
      expect(sanitizeFilename('')).toBe('upload');
      expect(sanitizeFilename('   ')).toBe('upload');
      expect(sanitizeFilename('...')).toBe('upload');
    });
  });
});
