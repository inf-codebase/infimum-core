import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-theme-toggle',
  standalone: true,
  imports: [CommonModule],
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
