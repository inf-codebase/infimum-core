import { Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ConfigService } from '../../../core/config';

/**
 * Home - Landing page with hero section, features, and CTA
 */
@Component({
  selector: 'app-home',
  imports: [RouterLink],
  templateUrl: './home.html',
  styleUrl: './home.css',
})
export class Home {
  configService = inject(ConfigService);

  get config() {
    return this.configService.getConfig();
  }

  features = [
    {
      icon: 'chart',
      title: 'Analytics Dashboard',
      description: 'Comprehensive analytics and reporting tools to track your business metrics.',
    },
    {
      icon: 'users',
      title: 'User Management',
      description: 'Easily manage users, roles, and permissions with our intuitive interface.',
    },
    {
      icon: 'shield',
      title: 'Enterprise Security',
      description: 'Bank-level security with encryption, 2FA, and compliance certifications.',
    },
    {
      icon: 'zap',
      title: 'Fast Performance',
      description: 'Optimized for speed with lazy loading and efficient caching strategies.',
    },
  ];
}
