import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { PageWrapper } from '../../../shared/components/page-wrapper/page-wrapper';
import { Card } from '../../../shared/components/card/card';
import { AppInput } from '../../../shared/components/app-input/app-input.component';
import { AppButton } from '../../../shared/components/app-button/app-button.component';
import { AppCheckbox } from '../../../shared/components/app-checkbox/app-checkbox.component';
import { ToastService } from '../../../shared/services/toast.service';

@Component({
  selector: 'app-admin-settings',
  imports: [CommonModule, FormsModule, PageWrapper, Card, AppInput, AppButton, AppCheckbox],
  templateUrl: './settings.html',
  styleUrl: './settings.css',
})
export class Settings {
  toastService = inject(ToastService);
  siteConfig = {
    name: 'Infimum',
    maintenanceMode: false,
    registrationOpen: true,
  };

  saveSettings() {
    this.toastService.show({ type: 'success', message: 'Settings saved successfully!' });
  }
}
