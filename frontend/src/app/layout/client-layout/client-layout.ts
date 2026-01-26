import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { ConfigService } from '../../core/config';
import { ThemeToggle } from '../../shared/components/theme-toggle/theme-toggle';

/**
 * ClientLayout - Layout for public-facing pages
 * Includes navbar and footer
 */
@Component({
  selector: 'app-client-layout',
  standalone: true,
  imports: [CommonModule, RouterOutlet, RouterLink, RouterLinkActive, ThemeToggle],
  templateUrl: './client-layout.html',
})
export class ClientLayout {
  configService = inject(ConfigService);
  mobileMenuOpen = false;

  get config() {
    return this.configService.getConfig();
  }

  get navItems() {
    return this.configService.getNavigation('public');
  }

  toggleMobileMenu() {
    this.mobileMenuOpen = !this.mobileMenuOpen;
  }
}
