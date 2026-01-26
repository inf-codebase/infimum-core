import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ToastService, ToastMessage } from '../../services/toast.service';

@Component({
    selector: 'app-toast',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './app-toast.component.html',
})
export class AppToast {
    toastService = inject(ToastService);

    getAlertClass(type: string): string {
        switch (type) {
            case 'success': return 'alert-success';
            case 'error': return 'alert-error';
            case 'warning': return 'alert-warning';
            default: return 'alert-info';
        }
    }
}
