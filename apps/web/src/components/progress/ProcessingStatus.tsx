'use client';

import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';
import type { Job } from '@rebloom/shared';

interface ProcessingStatusProps {
  job: Job;
  className?: string;
}

const steps = [
  { key: 'upload', label: 'Upload', threshold: 10 },
  { key: 'queue', label: 'Queue', threshold: 20 },
  { key: 'analyze', label: 'Analyze', threshold: 40 },
  { key: 'enhance', label: 'Enhance', threshold: 70 },
  { key: 'finalize', label: 'Done', threshold: 100 },
];

export function ProcessingStatus({ job, className }: ProcessingStatusProps) {
  const isProcessing = job.status === 'processing' || job.status === 'pending';
  const isFailed = job.status === 'failed';

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn('card p-6', className)}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-medium text-stone-900">
            {isFailed ? 'Processing failed' : isProcessing ? 'Enhancing...' : 'Complete'}
          </h3>
          <p className="text-sm text-stone-500">
            {isFailed 
              ? (job.error?.message || 'Something went wrong') 
              : isProcessing 
                ? 'This usually takes 10-15 seconds' 
                : 'Your image is ready'}
          </p>
        </div>
        <span className={cn(
          'text-sm font-mono',
          isFailed ? 'text-[#FF7F66]' : 'text-stone-600'
        )}>
          {job.progress}%
        </span>
      </div>

      {/* Progress Bar */}
      <div className="h-2 bg-stone-100 rounded-full overflow-hidden mb-6">
        <motion.div
          className={cn(
            'h-full rounded-full',
            isFailed 
              ? 'bg-[#FF7F66]' 
              : 'bg-gradient-to-r from-[#FF7F66] to-[#FF9B85]'
          )}
          initial={{ width: 0 }}
          animate={{ width: `${job.progress}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      </div>

      {/* Steps */}
      <div className="flex items-center justify-between">
        {steps.map((step, index) => {
          const isCompleted = job.progress >= step.threshold;
          const isActive = !isCompleted && (index === 0 || job.progress >= steps[index - 1].threshold);

          return (
            <div key={step.key} className="flex flex-col items-center gap-2">
              {/* Step dot */}
              <div className="relative">
                {/* Connection line */}
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      'absolute top-1/2 left-full h-px -translate-y-1/2',
                      isCompleted ? 'bg-[#FF9B85]' : 'bg-stone-200'
                    )}
                    style={{ width: 'calc(100% + 2.5rem)' }}
                  />
                )}

                {/* Dot */}
                <motion.div
                  className={cn(
                    'relative w-2.5 h-2.5 rounded-full',
                    isCompleted 
                      ? 'bg-[#FF7F66]' 
                      : isActive 
                        ? 'bg-[#FF9B85]' 
                        : 'bg-stone-200'
                  )}
                  animate={isActive && isProcessing ? { scale: [1, 1.3, 1] } : {}}
                  transition={{ repeat: Infinity, duration: 1.5 }}
                />
              </div>
              
              {/* Label */}
              <span
                className={cn(
                  'text-xs',
                  isCompleted 
                    ? 'text-stone-600' 
                    : isActive 
                      ? 'text-stone-500' 
                      : 'text-stone-300'
                )}
              >
                {step.label}
              </span>
            </div>
          );
        })}
      </div>

      {/* Model info */}
      <div className="mt-6 pt-4 border-t border-stone-100">
        <p className="text-xs text-stone-400">
          Using <span className="font-mono text-stone-500">{job.metadata.modelUsed}</span>
        </p>
      </div>
    </motion.div>
  );
}
