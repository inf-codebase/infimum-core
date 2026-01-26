import { Component, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './header.html',
  styleUrl: './header.css',
})
export class Header {
  toggleSidebar = output<void>();

  constructor(private router: Router) { }

  onLogout() {
    this.router.navigate(['/auth/login']);
  }
}
