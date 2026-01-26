import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PageWrapper } from '../../../shared/components/page-wrapper/page-wrapper';
import { Card } from '../../../shared/components/card/card';

@Component({
  selector: 'app-analytics',
  imports: [CommonModule, PageWrapper, Card],
  templateUrl: './analytics.html',
  styleUrl: './analytics.css',
})
export class Analytics {
  metrics = signal([
    { title: 'Page Views', value: '1.2M', change: '+12%', type: 'success' },
    { title: 'Unique Visitors', value: '850K', change: '+5%', type: 'success' },
    { title: 'Bounce Rate', value: '42%', change: '-2%', type: 'warning' },
    { title: 'Avg. Session', value: '4m 32s', change: '+30s', type: 'success' },
  ]);
}
