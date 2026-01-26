import { Component, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { ToolbarModule } from 'primeng/toolbar';
import { AvatarModule } from 'primeng/avatar';
import { ThemeToggle } from '../../shared/components/theme-toggle/theme-toggle';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, ButtonModule, ToolbarModule, AvatarModule, ThemeToggle],
  templateUrl: './header.html',
  styleUrl: './header.css',
})
export class Header {
  toggleSidebar = output<void>();
}
