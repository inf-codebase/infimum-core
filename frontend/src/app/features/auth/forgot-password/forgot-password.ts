import { Component, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AppInput } from '../../../shared/components/app-input/app-input.component';
import { AppButton } from '../../../shared/components/app-button/app-button.component';

@Component({
  selector: 'app-forgot-password',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterLink,
    AppInput,
    AppButton
  ],
  templateUrl: './forgot-password.html',
  styleUrl: './forgot-password.css',
})
export class ForgotPassword {
  private fb = inject(FormBuilder);
  private router = inject(Router);

  isLoading = signal(false);
  isSubmitted = signal(false);

  form = this.fb.group({
    email: ['', [Validators.required, Validators.email]]
  });

  onSubmit() {
    if (this.form.invalid) return;

    this.isLoading.set(true);

    setTimeout(() => {
      // Mock submit
      this.isSubmitted.set(true);
      this.isLoading.set(false);
    }, 1000);
  }
}
