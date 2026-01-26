import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AppToast } from '../../shared/components/app-toast/app-toast.component';

/**
 * BaseLayout - Root wrapper for the entire application
 * Provides theme context and global components like toast notifications
 */
@Component({
  selector: 'app-base-layout',
  imports: [RouterOutlet, AppToast],
  templateUrl: './base-layout.html',
  styleUrl: './base-layout.css',
})
export class BaseLayout { }
