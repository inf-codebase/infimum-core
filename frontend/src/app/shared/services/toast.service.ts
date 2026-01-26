import { Injectable, signal } from '@angular/core';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface ToastMessage {
    id: string;
    type: ToastType;
    title?: string;
    message: string;
    duration?: number;
}

@Injectable({
    providedIn: 'root'
})
export class ToastService {
    toasts = signal<ToastMessage[]>([]);

    show(toast: Omit<ToastMessage, 'id'>) {
        const id = Date.now().toString();
        const newToast = { ...toast, id, duration: toast.duration || 3000 };

        this.toasts.update(current => [...current, newToast]);

        if (newToast.duration > 0) {
            setTimeout(() => {
                this.remove(id);
            }, newToast.duration);
        }
    }

    remove(id: string) {
        this.toasts.update(current => current.filter(t => t.id !== id));
    }
}
