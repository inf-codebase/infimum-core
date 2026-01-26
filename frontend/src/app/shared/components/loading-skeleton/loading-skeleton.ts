import { Component } from '@angular/core';
import { SkeletonModule } from 'primeng/skeleton';

@Component({
  selector: 'app-loading-skeleton',
  standalone: true,
  imports: [SkeletonModule],
  templateUrl: './loading-skeleton.html',
  styleUrl: './loading-skeleton.css',
})
export class LoadingSkeleton {

}
