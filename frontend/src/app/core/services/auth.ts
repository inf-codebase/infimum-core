
import { Injectable, signal, computed, PLATFORM_ID, inject } from '@angular/core';
import { Router } from '@angular/router';
import { isPlatformBrowser } from '@angular/common';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  // Mock user state
  private currentUserSig = signal<{ id: string, name: string } | null>(null);

  // Computed signals
  currentUser = computed(() => this.currentUserSig());
  isAuthenticated = computed(() => !!this.currentUserSig());

  private platformId = inject(PLATFORM_ID);
  private router = inject(Router);

  constructor() {
    // Check local storage or token on init only in browser
    if (isPlatformBrowser(this.platformId)) {
      const storedUser = localStorage.getItem('user');
      if (storedUser) {
        this.currentUserSig.set(JSON.parse(storedUser));
      }
    }
  }

  login(credentials: any): boolean {
    // Mock login logic
    if (credentials.email === 'admin@example.com') {
      const user = { id: '1', name: 'Admin User' };
      this.currentUserSig.set(user);
      if (isPlatformBrowser(this.platformId)) {
        localStorage.setItem('user', JSON.stringify(user));
      }
      this.router.navigate(['/admin']);
      return true;
    }
    return false;
  }

  logout() {
    this.currentUserSig.set(null);
    if (isPlatformBrowser(this.platformId)) {
      localStorage.removeItem('user');
    }
    this.router.navigate(['/']);
  }
}

