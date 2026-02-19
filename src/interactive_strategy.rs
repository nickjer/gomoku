use std::cell::{Cell, RefCell};

use ratatui::Terminal;
use ratatui::backend::Backend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::board::Board;
use crate::offset::Offset;
use crate::position_id::PositionId;
use crate::stone::Stone;
use crate::strategy::Strategy;

/// An interactive strategy that uses a TUI for human input.
pub struct InteractiveStrategy<B: Backend> {
    terminal: RefCell<Terminal<B>>,
    cursor: Cell<PositionId>,
}

impl<B: Backend> InteractiveStrategy<B> {
    #[must_use]
    pub fn new(terminal: Terminal<B>) -> Self {
        Self {
            terminal: RefCell::new(terminal),
            cursor: Cell::new(PositionId::center()),
        }
    }

    fn render(&self, state: &Board, current_stone: Stone) {
        let cursor = self.cursor.get();

        self.terminal
            .borrow_mut()
            .draw(|frame| {
                let [board_area, status_area, help_area] = Layout::vertical([
                    Constraint::Fill(1),
                    Constraint::Length(2),
                    Constraint::Length(1),
                ])
                .areas(frame.area());

                render_board(frame, board_area, state, cursor);
                render_status(frame, status_area, current_stone);
                render_help(frame, help_area);
            })
            .expect("render failed");
    }

    fn handle_input(&self, state: &Board) -> Option<PositionId> {
        loop {
            if let Ok(Event::Key(key)) = event::read() {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                let cursor = self.cursor.get();

                match key.code {
                    KeyCode::Up | KeyCode::Char('k') => {
                        self.cursor.set(move_cursor(cursor, Offset::new(-1, 0)));
                        return None;
                    }
                    KeyCode::Down | KeyCode::Char('j') => {
                        self.cursor.set(move_cursor(cursor, Offset::new(1, 0)));
                        return None;
                    }
                    KeyCode::Left | KeyCode::Char('h') => {
                        self.cursor.set(move_cursor(cursor, Offset::new(0, -1)));
                        return None;
                    }
                    KeyCode::Right | KeyCode::Char('l') => {
                        self.cursor.set(move_cursor(cursor, Offset::new(0, 1)));
                        return None;
                    }
                    KeyCode::Enter | KeyCode::Char(' ') => {
                        if state.is_empty(cursor) {
                            return Some(cursor);
                        }
                    }
                    KeyCode::Char('q') | KeyCode::Esc => {
                        std::process::exit(0);
                    }
                    _ => {}
                }
            }
        }
    }
}

fn move_cursor(cursor: PositionId, offset: Offset) -> PositionId {
    cursor.offset(offset).unwrap_or(cursor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;
    use ratatui::backend::TestBackend;

    fn pos(row: usize, col: usize) -> PositionId {
        PositionId::from_position(Position::new(row, col))
    }

    #[test]
    fn render_board_shows_empty_cells() {
        let backend = TestBackend::new(29, 15);
        let mut terminal = Terminal::new(backend).unwrap();
        let state = Board::new();

        terminal
            .draw(|frame| render_board(frame, frame.area(), &state, PositionId::center()))
            .unwrap();

        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((0, 0)).unwrap().symbol(), "·");
    }

    #[test]
    fn render_board_shows_black_stone() {
        let backend = TestBackend::new(29, 15);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = Board::new();
        state.place(pos(0, 0), Stone::Black).unwrap();

        terminal
            .draw(|frame| render_board(frame, frame.area(), &state, PositionId::center()))
            .unwrap();

        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((0, 0)).unwrap().symbol(), "X");
    }

    #[test]
    fn render_board_shows_white_stone() {
        let backend = TestBackend::new(29, 15);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut state = Board::new();
        state.place(pos(0, 0), Stone::White).unwrap();

        terminal
            .draw(|frame| render_board(frame, frame.area(), &state, PositionId::center()))
            .unwrap();

        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.cell((0, 0)).unwrap().symbol(), "O");
    }

    #[test]
    fn render_status_shows_black_turn() {
        let backend = TestBackend::new(30, 1);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| render_status(frame, frame.area(), Stone::Black))
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = (0..30)
            .map(|x| buffer.cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(content.contains("Black (X)"));
    }

    #[test]
    fn render_status_shows_white_turn() {
        let backend = TestBackend::new(30, 1);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| render_status(frame, frame.area(), Stone::White))
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = (0..30)
            .map(|x| buffer.cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(content.contains("White (O)"));
    }

    #[test]
    fn render_help_shows_controls() {
        let backend = TestBackend::new(60, 1);
        let mut terminal = Terminal::new(backend).unwrap();

        terminal
            .draw(|frame| render_help(frame, frame.area()))
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content: String = (0..60)
            .map(|x| buffer.cell((x, 0)).unwrap().symbol())
            .collect();
        assert!(content.contains("Arrow keys"));
        assert!(content.contains("Enter"));
        assert!(content.contains("quit"));
    }

    #[test]
    fn move_cursor_up_from_center() {
        let center = PositionId::center();
        let new_pos = move_cursor(center, Offset::new(-1, 0));
        assert_eq!(new_pos.row(), center.row() - 1);
        assert_eq!(new_pos.col(), center.col());
    }

    #[test]
    fn move_cursor_down_from_center() {
        let center = PositionId::center();
        let new_pos = move_cursor(center, Offset::new(1, 0));
        assert_eq!(new_pos.row(), center.row() + 1);
        assert_eq!(new_pos.col(), center.col());
    }

    #[test]
    fn move_cursor_stays_at_top_edge() {
        let top = pos(0, 7);
        let new_pos = move_cursor(top, Offset::new(-1, 0));
        assert_eq!(new_pos, top);
    }

    #[test]
    fn move_cursor_stays_at_bottom_edge() {
        let bottom = pos(14, 7);
        let new_pos = move_cursor(bottom, Offset::new(1, 0));
        assert_eq!(new_pos, bottom);
    }

    #[test]
    fn move_cursor_stays_at_left_edge() {
        let left = pos(7, 0);
        let new_pos = move_cursor(left, Offset::new(0, -1));
        assert_eq!(new_pos, left);
    }

    #[test]
    fn move_cursor_stays_at_right_edge() {
        let right = pos(7, 14);
        let new_pos = move_cursor(right, Offset::new(0, 1));
        assert_eq!(new_pos, right);
    }
}

impl<B: Backend> Strategy for InteractiveStrategy<B> {
    fn choose_move(
        &self,
        current_stone: Stone,
        board: &Board,
        _rng: &mut fastrand::Rng,
    ) -> PositionId {
        loop {
            self.render(board, current_stone);
            if let Some(position_id) = self.handle_input(board) {
                return position_id;
            }
        }
    }

    fn label(&self) -> &'static str {
        "Human"
    }
}

fn render_board(frame: &mut ratatui::Frame, area: Rect, state: &Board, cursor: PositionId) {
    let lines: Vec<Line> = PositionId::rows()
        .map(|row| {
            let spans: Vec<Span> = row
                .into_iter()
                .enumerate()
                .flat_map(|(col_idx, position_id)| {
                    let stone = state.stone(position_id);
                    let is_cursor = position_id == cursor;

                    let ch = match stone {
                        Some(Stone::Black) => "X",
                        Some(Stone::White) => "O",
                        None => "·",
                    };

                    let style = if is_cursor {
                        Style::default()
                            .bg(Color::Yellow)
                            .fg(Color::Black)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        match stone {
                            Some(Stone::Black) => Style::default().fg(Color::Cyan),
                            Some(Stone::White) => Style::default().fg(Color::Magenta),
                            None => Style::default().fg(Color::DarkGray),
                        }
                    };

                    if col_idx > 0 {
                        vec![Span::raw(" "), Span::styled(ch, style)]
                    } else {
                        vec![Span::styled(ch, style)]
                    }
                })
                .collect();

            Line::from(spans)
        })
        .collect();

    let paragraph = Paragraph::new(lines);
    frame.render_widget(paragraph, area);
}

fn render_status(frame: &mut ratatui::Frame, area: Rect, current_stone: Stone) {
    let stone_name = match current_stone {
        Stone::Black => "Black (X)",
        Stone::White => "White (O)",
    };
    let text = format!("Your turn: {stone_name}");
    let paragraph = Paragraph::new(text).style(Style::default().fg(Color::Green));
    frame.render_widget(paragraph, area);
}

fn render_help(frame: &mut ratatui::Frame, area: Rect) {
    let text = "Arrow keys/hjkl: move | Enter/Space: place | q/Esc: quit";
    let paragraph = Paragraph::new(text).style(Style::default().fg(Color::DarkGray));
    frame.render_widget(paragraph, area);
}
