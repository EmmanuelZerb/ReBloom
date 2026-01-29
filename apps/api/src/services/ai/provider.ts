/**
 * Interface commune pour les providers AI
 * Permet de switcher facilement entre Replicate et un modèle custom
 */

import type { EnhanceOptions, AIProvider } from '@rebloom/shared';

// ============================================
// AI Provider Interface
// ============================================

export interface AIProviderResult {
  success: boolean;
  outputUrl?: string;
  outputBuffer?: Buffer;
  processingTimeMs: number;
  error?: string;
}

export interface IAIProvider {
  name: AIProvider;

  /**
   * Traite une image et retourne le résultat
   * @param imageUrl URL de l'image source (accessible publiquement) ou chemin local
   * @param options Options de traitement
   */
  enhance(imageUrl: string, options: EnhanceOptions): Promise<AIProviderResult>;

  /**
   * Vérifie si le provider est disponible
   */
  isAvailable(): Promise<boolean>;
}

// ============================================
// Provider Factory
// ============================================

import { ReplicateProvider } from './replicate';
import { config_ } from '../../lib/config';
import { aiLogger as logger } from '../../lib/logger';

let currentProvider: IAIProvider | null = null;

export function getAIProvider(): IAIProvider {
  if (currentProvider) return currentProvider;

  // TODO: Add support for custom model when enabled
  if (config_.customModel.enabled) {
    logger.info('Custom model enabled, but not implemented yet. Falling back to Replicate.');
  }

  currentProvider = new ReplicateProvider();
  logger.info({ provider: currentProvider.name }, 'AI provider initialized');

  return currentProvider;
}

/**
 * Permet de changer de provider à runtime (pour tests ou migration)
 */
export function setAIProvider(provider: IAIProvider): void {
  currentProvider = provider;
  logger.info({ provider: provider.name }, 'AI provider changed');
}
