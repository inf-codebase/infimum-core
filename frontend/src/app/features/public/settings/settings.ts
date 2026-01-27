import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Card } from '../../../shared/components/card/card';
import { AppButton } from '../../../shared/components/app-button/app-button.component';
import { AppCheckbox } from '../../../shared/components/app-checkbox/app-checkbox.component';
import { ThemeToggle } from '../../../shared/components/theme-toggle/theme-toggle';
import { ToastService } from '../../../shared/services/toast.service';

/**
 * Settings - User preferences and settings
 */
@Component({
  selector: 'app-settings',
  imports: [CommonModule, FormsModule, Card, AppButton, ThemeToggle],
  templateUrl: './settings.html',
  styleUrl: './settings.css',
})
export class Settings {
  toastService = inject(ToastService);

  settings = {
    emailNotifications: true,
    pushNotifications: false,
    marketingEmails: false,
    twoFactorAuth: true,
  };

  saveSettings() {
    this.toastService.show({ type: 'success', message: 'Settings saved successfully!' });
  }
}
