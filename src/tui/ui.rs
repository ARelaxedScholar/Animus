//! TUI Rendering

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, List, ListItem, Paragraph, Row, Table, Tabs, Wrap},
    Frame,
};

use crate::tui::app::{App, InputMode, Tab};

/// Render the entire UI
pub fn render(frame: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header/tabs
            Constraint::Min(0),    // Main content
            Constraint::Length(3), // Footer/status
        ])
        .split(frame.size());

    render_header(frame, app, chunks[0]);
    render_content(frame, app, chunks[1]);
    render_footer(frame, app, chunks[2]);
}

fn render_header(frame: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = Tab::all()
        .iter()
        .map(|t| {
            let style = if *t == app.active_tab {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(vec![
                Span::styled(
                    format!("{} ", t.key()),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(t.title(), style),
            ])
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Animus Dashboard "),
        )
        .highlight_style(Style::default().fg(Color::Yellow))
        .select(
            Tab::all()
                .iter()
                .position(|t| *t == app.active_tab)
                .unwrap_or(0),
        );

    frame.render_widget(tabs, area);
}

fn render_content(frame: &mut Frame, app: &App, area: Rect) {
    match app.active_tab {
        Tab::Dashboard => render_dashboard(frame, app, area),
        Tab::Videos => render_videos(frame, app, area),
        Tab::Queue => render_queue(frame, app, area),
        Tab::Retry => render_retry(frame, app, area),
        Tab::Settings => render_settings(frame, app, area),
    }
}

fn render_footer(frame: &mut Frame, app: &App, area: Rect) {
    let connection_status = if app.connected {
        Span::styled(
            " CONNECTED ",
            Style::default().bg(Color::Green).fg(Color::Black),
        )
    } else {
        Span::styled(
            " DISCONNECTED ",
            Style::default().bg(Color::Red).fg(Color::White),
        )
    };

    let daemon_status = if app.status.paused {
        Span::styled(
            " PAUSED ",
            Style::default().bg(Color::Yellow).fg(Color::Black),
        )
    } else if app.status.running {
        Span::styled(
            " RUNNING ",
            Style::default().bg(Color::Blue).fg(Color::White),
        )
    } else {
        Span::styled(
            " STOPPED ",
            Style::default().bg(Color::DarkGray).fg(Color::White),
        )
    };

    let busy_indicator = if app.busy {
        Span::styled(
            " BUSY ",
            Style::default().bg(Color::Magenta).fg(Color::White),
        )
    } else {
        Span::raw("")
    };

    let last_log = app
        .activity_log
        .last()
        .map(|s| {
            let content = if s.len() > 50 {
                format!("{}...", &s[..47])
            } else {
                s.clone()
            };
            Span::styled(
                format!(" | Last: {}", content),
                Style::default().fg(Color::Cyan),
            )
        })
        .unwrap_or_else(|| Span::raw(""));

    let help = match app.active_tab {
        Tab::Dashboard => "r: refresh | Tab: next tab | q: quit",
        Tab::Videos => "j/k: navigate | f: filter | d: download | r: refresh | q: quit",
        Tab::Queue => "a: add | d: delete | c: clear | j/k: navigate | q: quit",
        Tab::Retry => "Enter: retry selected | j/k: navigate | q: quit",
        Tab::Settings => "p: pause/resume | s: shutdown | q: quit",
    };

    let footer = Paragraph::new(Line::from(vec![
        connection_status,
        Span::raw(" "),
        daemon_status,
        Span::raw(" "),
        busy_indicator,
        last_log,
        Span::raw("  "),
        Span::styled(help, Style::default().fg(Color::DarkGray)),
    ]))
    .block(Block::default().borders(Borders::ALL));

    frame.render_widget(footer, area);
}

// =============================================================================
// Dashboard Tab
// =============================================================================

fn render_dashboard(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(8), Constraint::Min(0)])
        .split(chunks[0]);

    // Status panel
    let status_text = vec![
        Line::from(vec![
            Span::raw("Current Video: "),
            Span::styled(
                app.status.current_video_id.as_deref().unwrap_or("None"),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::raw("Current Stage: "),
            Span::styled(
                app.status.current_stage.as_deref().unwrap_or("-"),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("Next Scheduled: "),
            Span::styled(
                app.status
                    .next_scheduled_video
                    .as_deref()
                    .map(|s| s.split('T').next().unwrap_or(s))
                    .unwrap_or("-"),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::raw("Videos Produced: "),
            Span::styled(
                app.status.videos_produced.to_string(),
                Style::default().fg(Color::Magenta),
            ),
        ]),
    ];

    let status = Paragraph::new(status_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Current Status "),
    );
    frame.render_widget(status, left_chunks[0]);

    // Activity log
    let log_items: Vec<ListItem> = app
        .activity_log
        .iter()
        .rev()
        .take(20)
        .map(|s| ListItem::new(s.as_str()))
        .collect();

    let log = List::new(log_items).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Activity Log "),
    );
    frame.render_widget(log, left_chunks[1]);

    // Stats panel
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(10), Constraint::Min(0)])
        .split(chunks[1]);

    let stats_text = vec![
        Line::from(vec![
            Span::raw("Total Videos:   "),
            Span::styled(
                app.stats.total_videos.to_string(),
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            Span::raw("Published:      "),
            Span::styled(
                app.stats.published.to_string(),
                Style::default().fg(Color::Green),
            ),
        ]),
        Line::from(vec![
            Span::raw("Failed:         "),
            Span::styled(
                app.stats.failed.to_string(),
                Style::default().fg(Color::Red),
            ),
        ]),
        Line::from(vec![
            Span::raw("Producing:      "),
            Span::styled(
                app.stats.producing.to_string(),
                Style::default().fg(Color::Yellow),
            ),
        ]),
        Line::from(vec![
            Span::raw("Queue Length:   "),
            Span::styled(
                app.stats.queue_length.to_string(),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("Success Rate:   "),
            Span::styled(
                format!("{:.1}%", app.stats.success_rate),
                Style::default().fg(if app.stats.success_rate > 50.0 {
                    Color::Green
                } else {
                    Color::Red
                }),
            ),
        ]),
    ];

    let stats = Paragraph::new(stats_text)
        .block(Block::default().borders(Borders::ALL).title(" Statistics "));
    frame.render_widget(stats, right_chunks[0]);

    // Failure breakdown
    let failure_rows: Vec<Row> = app
        .stats
        .recent_failures
        .iter()
        .map(|f| {
            Row::new(vec![
                Cell::from(f.stage.clone()),
                Cell::from(f.count.to_string()),
            ])
        })
        .collect();

    let failures = Table::new(
        failure_rows,
        [Constraint::Percentage(70), Constraint::Percentage(30)],
    )
    .header(Row::new(vec!["Stage", "Count"]).style(Style::default().add_modifier(Modifier::BOLD)))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Failure Breakdown "),
    );

    frame.render_widget(failures, right_chunks[1]);
}

// =============================================================================
// Videos Tab
// =============================================================================

fn render_videos(frame: &mut Frame, app: &App, area: Rect) {
    let filter_text = match app.videos_filter.as_deref() {
        None => "All".to_string(),
        Some(f) => f.to_string(),
    };

    let rows: Vec<Row> = app
        .videos
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let style = if i == app.videos_selected {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            let status_style = match v.status.as_str() {
                "published" => Style::default().fg(Color::Green),
                "failed" => Style::default().fg(Color::Red),
                "producing" => Style::default().fg(Color::Yellow),
                _ => Style::default(),
            };

            Row::new(vec![
                Cell::from(v.id.chars().take(8).collect::<String>()),
                Cell::from(v.status.clone()).style(status_style),
                Cell::from(v.title.clone().unwrap_or_else(|| "-".to_string())),
                Cell::from(
                    v.created_at
                        .split('T')
                        .next()
                        .unwrap_or(&v.created_at)
                        .to_string(),
                ),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Length(12),
            Constraint::Min(20),
            Constraint::Length(12),
        ],
    )
    .header(
        Row::new(vec!["ID", "Status", "Title", "Created"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(format!(
        " Videos ({}) [f: filter, d: download] ",
        filter_text
    )));

    frame.render_widget(table, area);
}

// =============================================================================
// Queue Tab
// =============================================================================

fn render_queue(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(5)])
        .split(area);

    // Queue list
    let rows: Vec<Row> = app
        .queue
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let style = if i == app.queue_selected {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(item.id.to_string()),
                Cell::from(item.seed_topic.clone()),
                Cell::from(item.source_focus.clone().unwrap_or_else(|| "-".to_string())),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(6),
            Constraint::Min(30),
            Constraint::Length(20),
        ],
    )
    .header(
        Row::new(vec!["ID", "Topic", "Source"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Seed Queue ({} items) ", app.queue.len())),
    );

    frame.render_widget(table, chunks[0]);

    // Input area
    let input_style = if app.input_mode == InputMode::Editing {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let topic_style = if app.input_mode == InputMode::Editing && !app.queue_editing_source {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };

    let source_style = if app.input_mode == InputMode::Editing && app.queue_editing_source {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default()
    };

    let input_text = vec![
        Line::from(vec![
            Span::styled("Topic:  ", topic_style),
            Span::raw(&app.queue_input),
            if app.input_mode == InputMode::Editing && !app.queue_editing_source {
                Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK))
            } else {
                Span::raw("")
            },
        ]),
        Line::from(vec![
            Span::styled("Source: ", source_style),
            Span::raw(&app.queue_source_input),
            if app.input_mode == InputMode::Editing && app.queue_editing_source {
                Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK))
            } else {
                Span::raw("")
            },
        ]),
    ];

    let input_block = Paragraph::new(input_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Add Topic (a: start, Enter: submit, Esc: cancel, Tab: switch field) ")
            .border_style(input_style),
    );

    frame.render_widget(input_block, chunks[1]);
}

// =============================================================================
// Retry Tab
// =============================================================================

fn render_retry(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(4)])
        .split(area);

    // Retryable videos list
    let rows: Vec<Row> = app
        .retry_videos
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let style = if i == app.retry_selected {
                Style::default().bg(Color::DarkGray)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(v.id.chars().take(8).collect::<String>()),
                Cell::from(v.title.clone().unwrap_or_else(|| "-".to_string())),
                Cell::from(
                    v.error_message
                        .clone()
                        .unwrap_or_else(|| "-".to_string())
                        .chars()
                        .take(50)
                        .collect::<String>(),
                ),
            ])
            .style(style)
        })
        .collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(10),
            Constraint::Percentage(40),
            Constraint::Percentage(50),
        ],
    )
    .header(
        Row::new(vec!["ID", "Title", "Error"]).style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(format!(
        " Videos Failed at Publisher ({}) - Press Enter to retry ",
        app.retry_videos.len()
    )));

    frame.render_widget(table, chunks[0]);

    // Result panel
    let result_text = if app.retry_in_progress {
        vec![Line::from(Span::styled(
            "Retrying upload...",
            Style::default().fg(Color::Yellow),
        ))]
    } else {
        match &app.retry_result {
            Some(Ok(url)) => vec![Line::from(vec![
                Span::styled("Success! ", Style::default().fg(Color::Green)),
                Span::raw(url),
            ])],
            Some(Err(e)) => vec![Line::from(Span::styled(
                format!("Error: {}", e),
                Style::default().fg(Color::Red),
            ))],
            None => vec![Line::from(Span::styled(
                "Select a video and press Enter to retry",
                Style::default().fg(Color::DarkGray),
            ))],
        }
    };

    let result = Paragraph::new(result_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Retry Result "),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(result, chunks[1]);
}

// =============================================================================
// Settings Tab
// =============================================================================

fn render_settings(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(0)])
        .split(area);

    let status_color = if app.status.paused {
        Color::Yellow
    } else if app.status.running {
        Color::Green
    } else {
        Color::Red
    };

    let status_text = if app.status.paused {
        "PAUSED"
    } else if app.status.running {
        "RUNNING"
    } else {
        "STOPPED"
    };

    let settings_text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Daemon Status:  "),
            Span::styled(
                status_text,
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::raw("  Videos Produced: "),
            Span::styled(
                app.status.videos_produced.to_string(),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  [p] Toggle Pause/Resume",
            Style::default().fg(Color::White),
        )),
        Line::from(Span::styled(
            "  [s] Shutdown Daemon",
            Style::default().fg(Color::Red),
        )),
        Line::from(""),
    ];

    let settings = Paragraph::new(settings_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Daemon Control "),
    );

    frame.render_widget(settings, chunks[0]);

    // Last error
    let error_text = if let Some(ref err) = app.status.last_error {
        vec![Line::from(Span::styled(
            err,
            Style::default().fg(Color::Red),
        ))]
    } else {
        vec![Line::from(Span::styled(
            "No recent errors",
            Style::default().fg(Color::DarkGray),
        ))]
    };

    let errors = Paragraph::new(error_text)
        .block(Block::default().borders(Borders::ALL).title(" Last Error "))
        .wrap(Wrap { trim: true });

    frame.render_widget(errors, chunks[1]);
}
