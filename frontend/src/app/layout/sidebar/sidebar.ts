import { Component, input, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MenuModule } from 'primeng/menu';
import { MenuItem } from 'primeng/api';

@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, MenuModule],
  templateUrl: './sidebar.html',
  styleUrl: './sidebar.css',
})
export class Sidebar implements OnInit {
  isCollapsed = input(false);
  items: MenuItem[] | undefined;

  ngOnInit() {
    this.items = [
      {
        label: 'General',
        items: [
          {
            label: 'Dashboard',
            icon: 'pi pi-home',
            routerLink: '/dashboard'
          },
          {
            label: 'Settings',
            icon: 'pi pi-cog',
            routerLink: '/settings'
          }
        ]
      }
    ];
  }
}
