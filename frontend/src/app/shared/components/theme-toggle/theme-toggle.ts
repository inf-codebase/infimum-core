import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';

@Component({
  selector: 'app-theme-toggle',
  standalone: true,
  imports: [CommonModule, ButtonModule],
  templateUrl: './theme-toggle.html',
  styleUrl: './theme-toggle.css',
})
export class ThemeToggle {
  isDarkMode = signal(false);

  toggleTheme() {
    this.isDarkMode.update(v => !v);
    const element = document.querySelector('html');
    if (element) {
      element.classList.toggle('dark-mode');
    }
  }
}
