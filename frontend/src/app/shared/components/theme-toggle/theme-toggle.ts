import { Component, signal, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ConfigService } from '../../../core/config';

@Component({
  selector: 'app-theme-toggle',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './theme-toggle.html',
  styleUrl: './theme-toggle.css',
})
export class ThemeToggle {
  configService = inject(ConfigService);
  isDarkMode = signal(false);

  toggleTheme() {
    this.isDarkMode.update(v => !v);
    const lightTheme = this.configService.get('defaultTheme');
    const darkTheme = this.configService.get('darkTheme') || 'dark';

    // Fallback if config isn't loaded yet or property missing
    const theme = this.isDarkMode() ? darkTheme : lightTheme;
    document.documentElement.setAttribute('data-theme', theme);
  }
}
