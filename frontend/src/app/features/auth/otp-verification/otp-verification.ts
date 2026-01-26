import { Component, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService } from '../../../core/services/auth';
import { AppInput } from '../../../shared/components/app-input/app-input.component';
import { AppButton } from '../../../shared/components/app-button/app-button.component';

@Component({
  selector: 'app-otp-verification',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    AppInput,
    AppButton
  ],
  templateUrl: './otp-verification.html',
  styleUrl: './otp-verification.css',
})
export class OtpVerification {
  private fb = inject(FormBuilder);
  private authService = inject(AuthService);
  private router = inject(Router);

  isLoading = signal(false);

  // Mock email masking
  email = 'admin@example.com';

  form = this.fb.group({
    code: ['', [Validators.required, Validators.minLength(4)]]
  });

  onSubmit() {
    if (this.form.invalid) return;

    this.isLoading.set(true);

    setTimeout(() => {
      // Mock verify
      this.authService.login({ email: this.email });
      this.isLoading.set(false);
    }, 1500);
  }
}
