'use client';

import { motion } from 'framer-motion';
import type { Job } from '@rebloom/shared';

interface ProcessingStatusProps {
  job: Job;
}

export function ProcessingStatus({ job }: ProcessingStatusProps) {
  const isFailed = job.status === 'failed';

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-stone-600">
          {isFailed ? 'Failed' : 'Processing...'}
        </span>
        <span className="text-sm font-mono text-stone-500">{job.progress}%</span>
      </div>
      
      <div className="h-1.5 bg-stone-100 rounded-full overflow-hidden">
        <motion.div
          className={`h-full rounded-full ${isFailed ? 'bg-red-500' : 'bg-stone-900'}`}
          initial={{ width: 0 }}
          animate={{ width: `${job.progress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>

      {isFailed && job.error && (
        <p className="text-xs text-red-500 mt-2">{job.error.message}</p>
      )}
    </div>
  );
}
