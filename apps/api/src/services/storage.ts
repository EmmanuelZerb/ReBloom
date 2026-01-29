/**
 * Service de stockage abstrait
 * Supporte local et S3-compatible
 * Optimized Sharp configuration for memory efficiency
 */

import fs from 'fs/promises';
import path from 'path';
import { nanoid } from 'nanoid';
import sharp from 'sharp';
import { config_ } from '../lib/config';
import { storageLogger as logger } from '../lib/logger';
import type { ImageInfo, ImageMimeType, OutputFormat } from '@rebloom/shared';
import { MIME_TYPE_EXTENSIONS } from '@rebloom/shared';

// ============================================
// Sharp Configuration for Memory Efficiency
// ============================================

// Disable Sharp's built-in cache to reduce memory usage
// The cache can consume a lot of memory when processing many images
sharp.cache(false);

// Limit concurrent operations to prevent memory spikes
// Each Sharp operation can use significant memory for large images
sharp.concurrency(1);

// Configure Sharp's memory limits
// This helps prevent OOM errors on large images
sharp.simd(true); // Use SIMD for faster processing (less time = less memory in use)

logger.info(
  {
    cache: false,
    concurrency: 1,
    simd: true,
  },
  'Sharp configured for memory efficiency'
);

// ============================================
// Storage Interface
// ============================================

export interface StorageProvider {
  save(buffer: Buffer, filename: string): Promise<string>;
  get(filePath: string): Promise<Buffer>;
  delete(filePath: string): Promise<void>;
  getUrl(filePath: string): string;
  exists(filePath: string): Promise<boolean>;
}

// ============================================
// Local Storage Implementation
// ============================================

class LocalStorage implements StorageProvider {
  private basePath: string;

  constructor(basePath: string) {
    this.basePath = path.resolve(basePath);
    this.ensureDirectory();
  }

  private async ensureDirectory() {
    await fs.mkdir(this.basePath, { recursive: true });
    await fs.mkdir(path.join(this.basePath, 'originals'), { recursive: true });
    await fs.mkdir(path.join(this.basePath, 'processed'), { recursive: true });
    logger.debug({ path: this.basePath }, 'Storage directory initialized');
  }

  async save(buffer: Buffer, filename: string): Promise<string> {
    const filePath = path.join(this.basePath, filename);
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, buffer);
    logger.debug({ filePath, size: buffer.length }, 'File saved');
    return filename;
  }

  async get(filePath: string): Promise<Buffer> {
    const fullPath = path.join(this.basePath, filePath);
    return fs.readFile(fullPath);
  }

  async delete(filePath: string): Promise<void> {
    const fullPath = path.join(this.basePath, filePath);
    await fs.unlink(fullPath).catch(() => {});
    logger.debug({ filePath }, 'File deleted');
  }

  getUrl(filePath: string): string {
    // In development, serve from API
    return `/api/files/${filePath}`;
  }

  async exists(filePath: string): Promise<boolean> {
    try {
      const fullPath = path.join(this.basePath, filePath);
      await fs.access(fullPath);
      return true;
    } catch {
      return false;
    }
  }

  getFullPath(filePath: string): string {
    return path.join(this.basePath, filePath);
  }
}

// ============================================
// Storage Service
// ============================================

class StorageService {
  private provider: LocalStorage;

  constructor() {
    // TODO: Add S3 support based on config
    this.provider = new LocalStorage(config_.storage.localPath);
  }

  /**
   * Sauvegarde une image uploadée et extrait ses métadonnées
   */
  async saveOriginal(
    buffer: Buffer,
    originalName: string,
    mimeType: ImageMimeType
  ): Promise<ImageInfo> {
    const id = nanoid(12);
    const ext = MIME_TYPE_EXTENSIONS[mimeType];
    const filename = `originals/${id}${ext}`;

    // Extraire les dimensions avec Sharp
    const metadata = await sharp(buffer).metadata();
    if (!metadata.width || !metadata.height) {
      throw new Error('Could not read image dimensions');
    }

    // Sauvegarder
    await this.provider.save(buffer, filename);

    return {
      id,
      filename,
      originalName,
      mimeType,
      size: buffer.length,
      width: metadata.width,
      height: metadata.height,
      url: this.provider.getUrl(filename),
    };
  }

  /**
   * Sauvegarde une image traitée
   * Optimized for memory efficiency with sequential operations
   */
  async saveProcessed(
    buffer: Buffer,
    jobId: string,
    format: OutputFormat = 'png'
  ): Promise<ImageInfo> {
    const id = nanoid(12);
    const ext = `.${format}`;
    const filename = `processed/${id}${ext}`;

    // Get metadata first, then release the buffer reference
    const inputMetadata = await sharp(buffer, {
      limitInputPixels: 268402689, // ~16384 x 16384 max
      failOn: 'error',
    }).metadata();

    // Convertir au format souhaité avec Sharp
    // Using options to limit memory usage
    let processedBuffer: Buffer;
    const sharpInstance = sharp(buffer, {
      limitInputPixels: 268402689,
      failOn: 'error',
    });

    switch (format) {
      case 'jpeg':
        processedBuffer = await sharpInstance
          .jpeg({
            quality: 95,
            mozjpeg: true, // Better compression
          })
          .toBuffer();
        break;
      case 'webp':
        processedBuffer = await sharpInstance
          .webp({
            quality: 95,
            effort: 4, // Balanced compression speed
          })
          .toBuffer();
        break;
      default:
        processedBuffer = await sharpInstance
          .png({
            compressionLevel: 6, // Balanced compression
          })
          .toBuffer();
    }

    // Get output metadata (reuse dimensions from input if processing didn't change them)
    const outputWidth = inputMetadata.width || 0;
    const outputHeight = inputMetadata.height || 0;

    await this.provider.save(processedBuffer, filename);

    const mimeType: ImageMimeType = `image/${format}` as ImageMimeType;

    logger.debug(
      {
        id,
        format,
        inputSize: buffer.length,
        outputSize: processedBuffer.length,
        compressionRatio: (processedBuffer.length / buffer.length).toFixed(2),
      },
      'Image processed and saved'
    );

    return {
      id,
      filename,
      originalName: `enhanced_${jobId}${ext}`,
      mimeType,
      size: processedBuffer.length,
      width: outputWidth,
      height: outputHeight,
      url: this.provider.getUrl(filename),
    };
  }

  /**
   * Récupère un fichier
   */
  async getFile(filePath: string): Promise<Buffer> {
    return this.provider.get(filePath);
  }

  /**
   * Vérifie si un fichier existe
   */
  async exists(filePath: string): Promise<boolean> {
    return this.provider.exists(filePath);
  }

  /**
   * Supprime un fichier
   */
  async delete(filePath: string): Promise<void> {
    return this.provider.delete(filePath);
  }

  /**
   * Retourne le chemin complet d'un fichier (pour le worker)
   */
  getFullPath(filePath: string): string {
    return (this.provider as LocalStorage).getFullPath(filePath);
  }

  /**
   * Génère une URL de téléchargement
   */
  getDownloadUrl(filePath: string): string {
    return this.provider.getUrl(filePath);
  }
}

export const storage = new StorageService();
