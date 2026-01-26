import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';

import { ClientLayout } from './client-layout';

describe('ClientLayout', () => {
  let component: ClientLayout;
  let fixture: ComponentFixture<ClientLayout>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ClientLayout],
      providers: [provideRouter([])]
    })
      .compileComponents();

    fixture = TestBed.createComponent(ClientLayout);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
