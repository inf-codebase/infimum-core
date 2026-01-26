import { Component, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { AuthService } from '../../../core/services/auth';
import { AppInput } from '../../../shared/components/app-input/app-input.component';
import { AppButton } from '../../../shared/components/app-button/app-button.component';
import { AppCheckbox } from '../../../shared/components/app-checkbox/app-checkbox.component';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterLink,
    AppInput,
    AppButton,
    AppCheckbox
  ],
  templateUrl: './register.html',
  styleUrl: './register.css',
})
export class Register {
  private fb = inject(FormBuilder);
  private authService = inject(AuthService);
  private router = inject(Router);

  isLoading = signal(false);

  form = this.fb.group({
    name: ['', [Validators.required, Validators.minLength(2)]],
    email: ['', [Validators.required, Validators.email]],
    password: ['', [Validators.required, Validators.minLength(6)]],
    acceptTerms: [false, [Validators.requiredTrue]]
  });

  onSubmit() {
    if (this.form.invalid) return;

    this.isLoading.set(true);

    setTimeout(() => {
      // Mock registration
      this.authService.login({ email: 'admin@example.com' }); // Auto login for demo
      this.isLoading.set(false);
    }, 1500);
  }
}
