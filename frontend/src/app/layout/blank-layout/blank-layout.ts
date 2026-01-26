import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

/**
 * BlankLayout - Minimal layout for error pages (404, 500, etc.)
 * No navigation, just centered content
 */
@Component({
  selector: 'app-blank-layout',
  imports: [RouterOutlet],
  templateUrl: './blank-layout.html',
  styleUrl: './blank-layout.css',
})
export class BlankLayout { }
