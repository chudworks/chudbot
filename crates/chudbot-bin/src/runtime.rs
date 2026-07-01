use std::net::SocketAddr;

use chudbot_api::{MessagePlatformEvents, MessagePlatformRegistry};
use chudbot_bot::{BotRunOptions, BotRuntime, BotRuntimeTypes};
use chudbot_web::{WebRuntimeTypes, WebState};
use tokio::task::JoinError;
use tokio_util::sync::CancellationToken;

use crate::SHUTDOWN_GRACE_PERIOD;
use crate::errors::BinError;

/// Run fully constructed bot and web services under one process supervisor.
///
/// The binary owns the process-level lifecycle: start the bot and web server as
/// sibling tasks, let either an OS signal or an early service exit begin
/// shutdown, then join both tasks before returning an error to `main`.
pub async fn run_runtime_services<R>(
    bot: BotRuntime<R>,
    platform_events: impl MessagePlatformEvents<
        Error = <R::Platforms as MessagePlatformRegistry>::Error,
    > + Send
    + 'static,
    web: WebState<R>,
    listen: SocketAddr,
) -> Result<(), BinError>
where
    R: BotRuntimeTypes + WebRuntimeTypes + 'static,
{
    // One parent token fans shutdown out to both services. Child tokens let each
    // service observe cancellation without giving it ownership of the process
    // supervisor's trigger.
    let shutdown = CancellationToken::new();
    let bot_shutdown = shutdown.child_token();
    let web_shutdown = shutdown.child_token();

    // Start both long-running services before waiting for exit conditions. Each
    // task maps its domain error into the binary's top-level error type so the
    // join path can treat bot and web results uniformly.
    let mut bot_task = tokio::spawn(async move {
        bot.run_with_options(
            platform_events,
            bot_shutdown,
            BotRunOptions {
                drain_timeout: SHUTDOWN_GRACE_PERIOD,
            },
        )
        .await
        .map_err(BinError::Bot)
    });
    let mut web_task = tokio::spawn(async move {
        // Axum wants a shutdown future; the token is the shared process signal.
        chudbot_web::run_until_shutdown(web, listen, async move {
            web_shutdown.cancelled().await;
        })
        .await
        .map_err(BinError::Web)
    });

    // Race operator shutdown against either service exiting. In serve mode the
    // bot and web server are a pair, so an early exit from one starts graceful
    // shutdown for the other instead of leaving a partial process alive.
    let mut bot_result = None;
    let mut web_result = None;
    tokio::select! {
        _ = shutdown_signal() => {
            tracing::info!("process shutdown requested");
            shutdown.cancel();
        }
        result = &mut bot_task => {
            tracing::info!("bot service exited; shutting down remaining services");
            shutdown.cancel();
            bot_result = Some(join_service_result("bot", result));
        }
        result = &mut web_task => {
            tracing::info!("web service exited; shutting down remaining services");
            shutdown.cancel();
            web_result = Some(join_service_result("web", result));
        }
    }

    // Always join both tasks. Even when one service failed first, the remaining
    // service still gets the cancellation signal and a chance to drain cleanly
    // before any error is reported.
    let bot_result = match bot_result {
        Some(result) => result,
        None => join_service_result("bot", bot_task.await),
    };
    let web_result = match web_result {
        Some(result) => result,
        None => join_service_result("web", web_task.await),
    };
    bot_result?;
    web_result?;
    tracing::info!("all services stopped");
    Ok(())
}

/// Flatten a service task result into the binary error model.
///
/// `Ok(Err(_))` is an ordinary bot or web failure. `Err(JoinError)` means Tokio
/// could not return the service result because the task was cancelled or
/// panicked, so it is logged and wrapped as a supervisor-level failure.
fn join_service_result(
    task: &'static str,
    result: Result<Result<(), BinError>, JoinError>,
) -> Result<(), BinError> {
    match result {
        Ok(result) => result,
        Err(source) => {
            log_service_join_error(task, &source);
            Err(BinError::TaskJoin { task, source })
        }
    }
}

/// Log join failures with severity based on what Tokio reports.
fn log_service_join_error(service: &'static str, error: &JoinError) {
    if error.is_cancelled() {
        tracing::warn!(service, error = %error, "service task was cancelled");
    } else if error.is_panic() {
        tracing::error!(service, error = %error, "service task panicked");
    } else {
        tracing::warn!(service, error = %error, "service task join failed");
    }
}

/// Wait for an operator shutdown request.
///
/// SIGINT is portable through Tokio's Ctrl+C helper. SIGTERM is Unix-only, so
/// non-Unix builds use a pending future to keep the same select shape without
/// pretending that signal exists.
async fn shutdown_signal() {
    let ctrl_c = async {
        // If Tokio cannot install the process signal hook, continuing would
        // leave this supervisor without a reliable operator stop path.
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        // SIGTERM is the normal stop/restart signal from Unix service managers.
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => tracing::info!("received SIGINT"),
        () = terminate => tracing::info!("received SIGTERM"),
    }
}
