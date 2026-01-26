import { Component, signal, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { PageWrapper } from '../../../../shared/components/page-wrapper/page-wrapper';
import { Card } from '../../../../shared/components/card/card';
import { LoadingSkeleton } from '../../../../shared/components/loading-skeleton/loading-skeleton';

@Component({
  selector: 'app-dashboard-page',
  standalone: true,
  imports: [CommonModule, ButtonModule, PageWrapper, Card, LoadingSkeleton],
  templateUrl: './dashboard-page.html',
  styleUrl: './dashboard-page.css',
})
export class DashboardPage implements OnInit {
  isLoading = signal(true);
  stats = signal([
    { title: 'Total Users', value: '1,234', trend: '+12%', color: 'text-blue-500' },
    { title: 'Revenue', value: '$45,678', trend: '+8%', color: 'text-green-500' },
    { title: 'Active Sessions', value: '456', trend: '-3%', color: 'text-orange-500' },
    { title: 'Bounce Rate', value: '24%', trend: '-1%', color: 'text-purple-500' }
  ]);

  ngOnInit() {
    // Simulate loading
    setTimeout(() => {
      this.isLoading.set(false);
    }, 1500);
  }

  refresh() {
    this.isLoading.set(true);
    setTimeout(() => {
      this.isLoading.set(false);
    }, 1000);
  }
}
