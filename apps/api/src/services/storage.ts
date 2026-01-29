/**
 * Service de stockage abstrait
 * Supporte local et S3-compatible
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
   */
  async saveProcessed(
    buffer: Buffer,
    jobId: string,
    format: OutputFormat = 'png'
  ): Promise<ImageInfo> {
    const id = nanoid(12);
    const ext = `.${format}`;
    const filename = `processed/${id}${ext}`;

    // Convertir au format souhaité avec Sharp
    let processedBuffer: Buffer;
    const sharpInstance = sharp(buffer);

    switch (format) {
      case 'jpeg':
        processedBuffer = await sharpInstance.jpeg({ quality: 95 }).toBuffer();
        break;
      case 'webp':
        processedBuffer = await sharpInstance.webp({ quality: 95 }).toBuffer();
        break;
      default:
        processedBuffer = await sharpInstance.png().toBuffer();
    }

    const metadata = await sharp(processedBuffer).metadata();

    await this.provider.save(processedBuffer, filename);

    const mimeType: ImageMimeType = `image/${format}` as ImageMimeType;

    return {
      id,
      filename,
      originalName: `enhanced_${jobId}${ext}`,
      mimeType,
      size: processedBuffer.length,
      width: metadata.width || 0,
      height: metadata.height || 0,
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
