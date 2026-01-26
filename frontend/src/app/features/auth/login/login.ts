import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../../core/services/auth';
import { AppInput } from '../../../shared/components/app-input/app-input.component';
import { AppButton } from '../../../shared/components/app-button/app-button.component';
import { AppCheckbox } from '../../../shared/components/app-checkbox/app-checkbox.component';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterLink,
    AppInput,
    AppButton,
    AppCheckbox
  ],
  templateUrl: './login.html',
  styleUrl: './login.css',
})
export class Login {
  private fb = inject(FormBuilder);
  private authService = inject(AuthService);
  private router = inject(Router);

  isLoading = signal(false);
  error = signal<string | null>(null);

  form = this.fb.group({
    email: ['admin@example.com', [Validators.required, Validators.email]],
    password: ['admin123', [Validators.required]],
    rememberMe: [false]
  });

  onSubmit() {
    if (this.form.invalid) return;

    this.isLoading.set(true);
    this.error.set(null);

    setTimeout(() => {
      const success = this.authService.login(this.form.value);
      if (!success) {
        this.error.set('Invalid email or password');
      }
      this.isLoading.set(false);
    }, 1000);
  }
}
