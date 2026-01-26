import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ButtonModule } from 'primeng/button';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-client-layout',
  standalone: true,
  imports: [RouterOutlet, ButtonModule, CommonModule],
  templateUrl: './client-layout.html',
  styleUrl: './client-layout.css',
})
export class ClientLayout {

}
