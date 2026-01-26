import { Component, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { CommonModule } from '@angular/common';
import { InputTextModule } from 'primeng/inputtext';
import { PasswordModule } from 'primeng/password';
import { CheckboxModule } from 'primeng/checkbox';
import { ButtonModule } from 'primeng/button';
import { AuthService } from '../../../../core/services/auth';
import { MessageModule } from 'primeng/message';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    RouterLink,
    InputTextModule,
    PasswordModule,
    CheckboxModule,
    ButtonModule,
    MessageModule
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
