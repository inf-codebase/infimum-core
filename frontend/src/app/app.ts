import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { AppToast } from './shared/components/app-toast/app-toast.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, AppToast],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('frontend');
}
