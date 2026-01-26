import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BlankLayout } from './blank-layout';

describe('BlankLayout', () => {
  let component: BlankLayout;
  let fixture: ComponentFixture<BlankLayout>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BlankLayout]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BlankLayout);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
